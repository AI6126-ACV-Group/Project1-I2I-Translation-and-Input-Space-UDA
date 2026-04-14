import argparse
import os
import time

import torch
import torch.nn as nn

from office31_uda.common import (
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
    save_progress,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CyCADA-style feature adaptation for Office-31 Amazon2Webcam -> Webcam."
    )
    parser.add_argument("--translated-root", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--source-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/cycada_feature/amazon2webcam_to_webcam")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-target", type=float, default=1e-6)
    parser.add_argument("--lr-discriminator", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lambda-adv", type=float, default=0.3)
    parser.add_argument("--disc-acc-threshold", type=float, default=0.6)
    parser.add_argument("--diagnostic-target-eval", action="store_true")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def load_source_weights(checkpoint_path: str, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "modules" in checkpoint and "model" in checkpoint["modules"]:
        model.load_state_dict(checkpoint["modules"]["model"])
    else:
        model.load_state_dict(checkpoint)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    setup_logging(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = create_writer(args.output_dir)
    print(f"Using device: {device}")
    print(f"Translated root: {args.translated_root}")
    print(f"Split dir: {args.split_dir}")
    print(f"Source checkpoint: {args.source_checkpoint}")
    print(f"Output dir: {args.output_dir}")

    loaders = build_translated_loaders(
        translated_root=args.translated_root,
        split_dir=args.split_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
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

            epoch_disc_loss = 0.0
            epoch_adv_loss = 0.0
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
                domain_logits = discriminator(
                    torch.cat([src_features.detach(), tgt_features.detach()], dim=0)
                )
                domain_labels = torch.cat(
                    [
                        torch.ones(src_features.size(0), dtype=torch.long, device=device),
                        torch.zeros(tgt_features.size(0), dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                disc_loss = criterion_domain(domain_logits, domain_labels)
                disc_loss.backward()
                optimizer_discriminator.step()

                disc_preds = domain_logits.argmax(dim=1)
                disc_acc = (disc_preds == domain_labels).float().mean().item()

                adv_loss_value = 0.0
                if disc_acc > args.disc_acc_threshold:
                    optimizer_target.zero_grad()
                    _, tgt_features = target_model(tgt_images, return_features=True)
                    fooled_logits = discriminator(tgt_features)
                    fooled_labels = torch.ones(tgt_features.size(0), dtype=torch.long, device=device)
                    adv_loss = criterion_domain(fooled_logits, fooled_labels) * args.lambda_adv
                    adv_loss.backward()
                    optimizer_target.step()
                    adv_loss_value = adv_loss.item()
                    epoch_target_updates += 1

                epoch_disc_loss += disc_loss.item()
                epoch_adv_loss += adv_loss_value
                epoch_disc_acc += disc_acc

            scheduler_target.step()
            scheduler_discriminator.step()

            proxy_val_loss, proxy_val_acc = evaluate_classifier(
                model=target_model,
                loader=loaders["translated_val"],
                criterion=criterion_cls,
                device=device,
            )

            epoch_log = {
                "epoch": epoch,
                "disc_loss": epoch_disc_loss / args.steps_per_epoch,
                "adv_loss": epoch_adv_loss / args.steps_per_epoch,
                "disc_acc": epoch_disc_acc / args.steps_per_epoch,
                "target_update_rate": epoch_target_updates / args.steps_per_epoch,
                "proxy_val_loss": proxy_val_loss,
                "proxy_val_acc": proxy_val_acc,
                "lr_target": optimizer_target.param_groups[0]["lr"],
                "lr_discriminator": optimizer_discriminator.param_groups[0]["lr"],
            }
            if args.diagnostic_target_eval:
                target_diag_loss, target_diag_acc = evaluate_classifier(
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

            save_progress(
                args.output_dir,
                history=history,
                summary={"best_proxy_acc": best_proxy_acc, "latest_epoch": epoch},
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

    proxy_val_loss, proxy_val_acc = evaluate_classifier(
        model=target_model,
        loader=loaders["translated_val"],
        criterion=criterion_cls,
        device=device,
    )
    target_test_loss, target_test_acc = evaluate_classifier(
        model=target_model,
        loader=loaders["target_test"],
        criterion=criterion_cls,
        device=device,
    )

    metrics = {
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

    save_json(os.path.join(args.output_dir, "metrics.json"), {"history": history, "metrics": metrics})

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Proxy validation accuracy: {proxy_val_acc * 100:.2f}%")
    print(f"Target test accuracy: {target_test_acc * 100:.2f}%")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
