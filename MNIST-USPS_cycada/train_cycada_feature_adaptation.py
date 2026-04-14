import argparse
import os
import time

omp_threads = os.environ.get("OMP_NUM_THREADS")
if omp_threads is not None:
    try:
        if int(omp_threads) <= 0:
            raise ValueError
    except ValueError:
        os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn

from digit_uda.common import (
    NUM_CLASSES,
    DomainDiscriminator,
    ForeverDataIterator,
    ResNet18FeatureNet,
    build_translated_loaders,
    create_writer,
    ensure_dir,
    evaluate_classifier,
    freeze_module,
    load_checkpoint,
    log_scalars,
    plot_history,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CyCADA-style feature adaptation from translated source to target."
    )
    parser.add_argument("--translated-root", type=str, default="./mnist2usps.zip")
    parser.add_argument("--target", type=str, default="usps", choices=["mnist", "usps"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--source-checkpoint", type=str, default="", help="Checkpoint from translated-source baseline.")
    parser.add_argument("--output-dir", type=str, default="./outputs/cycada_feature/mnist_as_usps_to_usps")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-target", type=float, default=1e-6)
    parser.add_argument("--lr-discriminator", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    parser.add_argument("--disc-acc-threshold", type=float, default=0.6)
    parser.add_argument("--diagnostic-target-eval", action="store_true")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def load_source_weights(checkpoint_path: str, model: nn.Module, device: torch.device) -> None:
    if not checkpoint_path:
        raise ValueError("CyCADA feature adaptation requires --source-checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "modules" in checkpoint and "model" in checkpoint["modules"]:
        model.load_state_dict(checkpoint["modules"]["model"])
    else:
        model.load_state_dict(checkpoint)


def evaluate_target_model(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    return evaluate_classifier(model=model, loader=loader, criterion=criterion, device=device)


def resolve_project_path(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(project_root, path))


def resolve_data_root(data_root: str, project_root: str, target: str) -> str:
    requested_root = resolve_project_path(data_root, project_root)
    project_data_root = os.path.join(project_root, "data")

    requested_target_dir = os.path.join(requested_root, target)
    project_target_dir = os.path.join(project_data_root, target)

    if os.path.isdir(requested_target_dir):
        return requested_root
    if os.path.isdir(project_target_dir):
        print(
            f"Using project-local data root instead of missing target dataset path: {project_data_root}"
        )
        return project_data_root
    return requested_root


def resolve_source_checkpoint(checkpoint_path: str, project_root: str) -> str:
    requested_path = resolve_project_path(checkpoint_path, project_root)
    if os.path.isfile(requested_path):
        return requested_path

    candidates = [
        os.path.join(project_root, "outputs", "translated_source", "mnist_as_usps", os.path.basename(checkpoint_path)),
        os.path.join(project_root, "outputs", "translated_source", "mnist_as_usps", "best.pth"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            print(f"Using fallback source checkpoint: {candidate}")
            return candidate

    raise FileNotFoundError(
        "Source checkpoint not found. Checked: "
        f"{requested_path} and {', '.join(candidates)}"
    )


def main() -> None:
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    args.translated_root = resolve_project_path(args.translated_root, project_root)
    args.output_dir = resolve_project_path(args.output_dir, project_root)
    args.data_root = resolve_data_root(args.data_root, project_root, args.target)
    if args.resume:
        args.resume = resolve_project_path(args.resume, project_root)
    if args.source_checkpoint:
        args.source_checkpoint = resolve_source_checkpoint(args.source_checkpoint, project_root)

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = create_writer(args.output_dir)

    print(f"Using device: {device}")
    print(f"Translated source: {args.translated_root}")
    print(f"Target: {args.target}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")

    loaders = build_translated_loaders(
        translated_root=args.translated_root,
        target=args.target,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        pretrained=args.pretrained,
        seed=args.seed,
    )

    source_model = ResNet18FeatureNet(num_classes=NUM_CLASSES, pretrained=args.pretrained).to(device)
    target_model = ResNet18FeatureNet(num_classes=NUM_CLASSES, pretrained=args.pretrained).to(device)
    discriminator = DomainDiscriminator(in_dim=source_model.feature_dim).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    optimizer_target = torch.optim.Adam(
        target_model.encoder.parameters(),
        lr=args.lr_target,
        weight_decay=args.weight_decay,
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_discriminator,
        weight_decay=args.weight_decay,
    )
    scheduler_target = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_target, T_max=args.epochs)
    scheduler_discriminator = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_discriminator, T_max=args.epochs
    )

    best_path = os.path.join(args.output_dir, "best.pth")
    last_path = os.path.join(args.output_dir, "last.pth")
    history = []
    best_proxy_acc = -1.0
    start_epoch = 1

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            modules={
                "source_model": source_model,
                "target_model": target_model,
                "discriminator": discriminator,
            },
            optimizers={
                "optimizer_target": optimizer_target,
                "optimizer_discriminator": optimizer_discriminator,
            },
            schedulers={
                "scheduler_target": scheduler_target,
                "scheduler_discriminator": scheduler_discriminator,
            },
            map_location=device,
        )
        state = checkpoint.get("state", {})
        history = state.get("history", [])
        best_proxy_acc = state.get("best_proxy_acc", best_proxy_acc)
        start_epoch = state.get("epoch", 0) + 1
        print(f"Resumed from: {args.resume} (next epoch: {start_epoch})")
    else:
        load_source_weights(args.source_checkpoint, source_model, device)
        target_model.load_state_dict(source_model.state_dict())

    freeze_module(source_model)
    freeze_module(target_model.classifier)

    if not args.eval_only:
        src_iter = ForeverDataIterator(loaders["translated_train"])
        tgt_iter = ForeverDataIterator(loaders["target_train"])
        train_start = time.time()

        for epoch in range(start_epoch, args.epochs + 1):
            target_model.train()
            target_model.classifier.eval()
            discriminator.train()

            epoch_loss_disc = 0.0
            epoch_loss_adv = 0.0
            epoch_disc_acc = 0.0
            epoch_target_updates = 0

            for _ in range(args.steps_per_epoch):
                src_images, _ = next(src_iter)
                tgt_images, _ = next(tgt_iter)
                src_images = src_images.to(device, non_blocking=True)
                tgt_images = tgt_images.to(device, non_blocking=True)

                with torch.no_grad():
                    _, src_features = source_model(src_images, return_features=True)

                _, tgt_features = target_model(tgt_images, return_features=True)

                optimizer_discriminator.zero_grad()
                domain_logits = discriminator(torch.cat([src_features.detach(), tgt_features.detach()], dim=0))
                domain_labels = torch.cat(
                    [
                        torch.ones(src_features.size(0), dtype=torch.long, device=device),
                        torch.zeros(tgt_features.size(0), dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                loss_disc = criterion_domain(domain_logits, domain_labels)
                loss_disc.backward()
                optimizer_discriminator.step()

                preds_disc = domain_logits.argmax(dim=1)
                acc_disc = (preds_disc == domain_labels).float().mean().item()

                loss_adv_value = 0.0
                if acc_disc > args.disc_acc_threshold:
                    optimizer_target.zero_grad()
                    _, tgt_features = target_model(tgt_images, return_features=True)
                    fooled_logits = discriminator(tgt_features)
                    fooled_labels = torch.ones(tgt_features.size(0), dtype=torch.long, device=device)
                    loss_adv = criterion_domain(fooled_logits, fooled_labels) * args.lambda_adv
                    loss_adv.backward()
                    optimizer_target.step()
                    loss_adv_value = loss_adv.item()
                    epoch_target_updates += 1

                epoch_loss_disc += loss_disc.item()
                epoch_loss_adv += loss_adv_value
                epoch_disc_acc += acc_disc

            scheduler_target.step()
            scheduler_discriminator.step()

            proxy_val_loss, proxy_val_acc = evaluate_target_model(
                model=target_model,
                loader=loaders["translated_val"],
                criterion=criterion_cls,
                device=device,
            )

            epoch_log = {
                "epoch": epoch,
                "disc_loss": epoch_loss_disc / args.steps_per_epoch,
                "adv_loss": epoch_loss_adv / args.steps_per_epoch,
                "disc_acc": epoch_disc_acc / args.steps_per_epoch,
                "target_update_rate": epoch_target_updates / args.steps_per_epoch,
                "proxy_val_loss": proxy_val_loss,
                "proxy_val_acc": proxy_val_acc,
                "lr_target": optimizer_target.param_groups[0]["lr"],
                "lr_discriminator": optimizer_discriminator.param_groups[0]["lr"],
            }
            if args.diagnostic_target_eval:
                target_diag_loss, target_diag_acc = evaluate_target_model(
                    model=target_model,
                    loader=loaders["target_test"],
                    criterion=criterion_cls,
                    device=device,
                )
                epoch_log["target_test_loss"] = target_diag_loss
                epoch_log["target_test_acc"] = target_diag_acc
            history.append(epoch_log)
            log_scalars(writer, epoch_log, epoch)

            log_message = (
                f"Epoch [{epoch}/{args.epochs}] "
                f"disc_loss={epoch_log['disc_loss']:.4f} "
                f"adv_loss={epoch_log['adv_loss']:.4f} "
                f"disc_acc={epoch_log['disc_acc']:.4f} "
                f"target_update_rate={epoch_log['target_update_rate']:.4f} "
                f"proxy_val_acc={proxy_val_acc:.4f}"
            )
            if "target_test_acc" in epoch_log:
                log_message += f" target_test_acc={epoch_log['target_test_acc']:.4f}"
            print(log_message)

            state = {
                "epoch": epoch,
                "history": history,
                "best_proxy_acc": best_proxy_acc,
                "args": vars(args),
                "elapsed_seconds": time.time() - train_start,
            }
            save_checkpoint(
                last_path,
                modules={
                    "source_model": source_model,
                    "target_model": target_model,
                    "discriminator": discriminator,
                },
                optimizers={
                    "optimizer_target": optimizer_target,
                    "optimizer_discriminator": optimizer_discriminator,
                },
                schedulers={
                    "scheduler_target": scheduler_target,
                    "scheduler_discriminator": scheduler_discriminator,
                },
                state=state,
            )
            if args.save_epoch_checkpoints:
                save_checkpoint(
                    os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth"),
                    modules={
                        "source_model": source_model,
                        "target_model": target_model,
                        "discriminator": discriminator,
                    },
                    optimizers={
                        "optimizer_target": optimizer_target,
                        "optimizer_discriminator": optimizer_discriminator,
                    },
                    schedulers={
                        "scheduler_target": scheduler_target,
                        "scheduler_discriminator": scheduler_discriminator,
                    },
                    state=state,
                )

            if proxy_val_acc > best_proxy_acc:
                best_proxy_acc = proxy_val_acc
                state["best_proxy_acc"] = best_proxy_acc
                save_checkpoint(
                    best_path,
                    modules={
                        "source_model": source_model,
                        "target_model": target_model,
                        "discriminator": discriminator,
                    },
                    optimizers={
                        "optimizer_target": optimizer_target,
                        "optimizer_discriminator": optimizer_discriminator,
                    },
                    schedulers={
                        "scheduler_target": scheduler_target,
                        "scheduler_discriminator": scheduler_discriminator,
                    },
                    state=state,
                )

    eval_path = best_path if os.path.exists(best_path) else last_path
    checkpoint = load_checkpoint(
        eval_path,
        modules={
            "source_model": source_model,
            "target_model": target_model,
            "discriminator": discriminator,
        },
        map_location=device,
    )
    best_proxy_acc = checkpoint.get("state", {}).get("best_proxy_acc", best_proxy_acc)
    history = checkpoint.get("state", {}).get("history", history)

    proxy_val_loss, proxy_val_acc = evaluate_target_model(
        model=target_model,
        loader=loaders["translated_val"],
        criterion=criterion_cls,
        device=device,
    )
    target_test_loss, target_test_acc = evaluate_target_model(
        model=target_model,
        loader=loaders["target_test"],
        criterion=criterion_cls,
        device=device,
    )

    metrics = {
        "translated_root": args.translated_root,
        "target": args.target,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "lr_target": args.lr_target,
        "lr_discriminator": args.lr_discriminator,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "pretrained": args.pretrained,
        "lambda_adv": args.lambda_adv,
        "disc_acc_threshold": args.disc_acc_threshold,
        "diagnostic_target_eval": args.diagnostic_target_eval,
        "best_proxy_acc": best_proxy_acc,
        "proxy_val_loss": proxy_val_loss,
        "proxy_val_acc": proxy_val_acc,
        "target_test_loss": target_test_loss,
        "target_test_acc": target_test_acc,
        "selection_checkpoint": eval_path,
        "metric": "top1_accuracy",
    }

    save_json(
        os.path.join(args.output_dir, "metrics.json"),
        {"history": history, "metrics": metrics},
    )
    plot_history(
        history,
        args.output_dir,
        groups={
            "loss": ["disc_loss", "adv_loss", "proxy_val_loss"],
            "accuracy": ["disc_acc", "target_update_rate", "proxy_val_acc", "target_test_acc"],
            "learning_rate": ["lr_target", "lr_discriminator"],
        },
    )

    if writer is not None:
        writer.close()

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Proxy validation accuracy: {proxy_val_acc * 100:.2f}%")
    print(f"Target test accuracy ({args.target}): {target_test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
