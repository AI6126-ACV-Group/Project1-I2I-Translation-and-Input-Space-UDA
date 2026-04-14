import argparse
import os
import time

import torch
import torch.nn as nn

from FDA.common import (
    ResNet18FeatureNet,
    create_writer,
    ensure_dir,
    evaluate_classifier,
    load_checkpoint,
    log_scalars,
    plot_history,
    save_checkpoint,
    save_json,
    save_progress,
    set_seed,
    setup_logging,
    train_one_epoch,
)
from FDA.data import TASKS, build_fda_training_loaders, get_task_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FDA classifier on translated source images.")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASKS))
    parser.add_argument("--data-root", type=str, default="./Datasets")
    parser.add_argument("--split-dir", type=str, default="")
    parser.add_argument("--fda-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--train-augment",
        type=str,
        default="default",
        choices=("default", "resize_flip"),
        help="Training augmentation for FDA source images.",
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = get_task_config(args.task)
    set_seed(args.seed)

    split_dir = args.split_dir or os.path.join("./FDA/outputs/splits", task.name)
    fda_root = args.fda_root or os.path.join("./FDA/outputs/exports", task.name)
    output_dir = args.output_dir or os.path.join("./FDA/outputs/runs", task.name)
    image_size = args.image_size or task.default_image_size

    ensure_dir(output_dir)
    setup_logging(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = create_writer(output_dir)

    print(f"Using device: {device}")
    print(f"Task: {task.name}")
    print(f"Data root: {os.path.abspath(args.data_root)}")
    print(f"Split dir: {os.path.abspath(split_dir)}")
    print(f"FDA root: {os.path.abspath(fda_root)}")
    print(f"Output dir: {os.path.abspath(output_dir)}")
    print(f"Train augment: {args.train_augment}")

    loaders = build_fda_training_loaders(
        task=task,
        data_root=args.data_root,
        split_dir=split_dir,
        fda_root=fda_root,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        train_augment=args.train_augment,
    )

    model = ResNet18FeatureNet(num_classes=task.num_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_path = os.path.join(output_dir, "best.pth")
    last_path = os.path.join(output_dir, "last.pth")
    history = []
    best_val_acc = -1.0
    start_epoch = 1

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            modules={"model": model},
            optimizers={"optimizer": optimizer},
            schedulers={"scheduler": scheduler},
            map_location=device,
        )
        state = checkpoint.get("state", {})
        history = state.get("history", [])
        best_val_acc = state.get("best_val_acc", best_val_acc)
        start_epoch = state.get("epoch", 0) + 1
        print(f"Resumed from: {args.resume} (next epoch: {start_epoch})")

    if not args.eval_only:
        start_time = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=loaders["fda_train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            val_loss, val_acc = evaluate_classifier(
                model=model,
                loader=loaders["fda_val"],
                criterion=criterion,
                device=device,
            )
            scheduler.step()

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_log)
            log_scalars(writer, epoch_log, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            state = {
                "epoch": epoch,
                "history": history,
                "best_val_acc": best_val_acc,
                "args": vars(args),
                "task": task.name,
                "elapsed_seconds": time.time() - start_time,
            }
            save_checkpoint(
                last_path,
                modules={"model": model},
                optimizers={"optimizer": optimizer},
                schedulers={"scheduler": scheduler},
                state=state,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                state["best_val_acc"] = best_val_acc
                save_checkpoint(
                    best_path,
                    modules={"model": model},
                    optimizers={"optimizer": optimizer},
                    schedulers={"scheduler": scheduler},
                    state=state,
                )

            save_progress(
                output_dir,
                history=history,
                summary={"best_val_acc": best_val_acc, "latest_epoch": epoch},
            )
            plot_history(
                history,
                output_dir,
                groups={
                    "loss": ["train_loss", "val_loss"],
                    "accuracy": ["train_acc", "val_acc"],
                    "learning_rate": ["lr"],
                },
            )

    if args.eval_only and args.resume:
        eval_path = args.resume
    else:
        eval_path = best_path if os.path.exists(best_path) else last_path
    checkpoint = load_checkpoint(eval_path, modules={"model": model}, map_location=device)
    best_val_acc = checkpoint.get("state", {}).get("best_val_acc", best_val_acc)
    history = checkpoint.get("state", {}).get("history", history)

    fda_val_loss, fda_val_acc = evaluate_classifier(
        model=model,
        loader=loaders["fda_val"],
        criterion=criterion,
        device=device,
    )
    target_test_loss, target_test_acc = evaluate_classifier(
        model=model,
        loader=loaders["target_test"],
        criterion=criterion,
        device=device,
    )

    metrics = {
        "task": task.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": image_size,
        "pretrained": args.pretrained,
        "train_augment": args.train_augment,
        "best_val_acc": best_val_acc,
        "fda_val_loss": fda_val_loss,
        "fda_val_acc": fda_val_acc,
        "target_test_loss": target_test_loss,
        "target_test_acc": target_test_acc,
        "selection_checkpoint": eval_path,
        "metric": "top1_accuracy",
    }

    if "source_test" in loaders:
        source_test_loss, source_test_acc = evaluate_classifier(
            model=model,
            loader=loaders["source_test"],
            criterion=criterion,
            device=device,
        )
        metrics["source_test_loss"] = source_test_loss
        metrics["source_test_acc"] = source_test_acc

    save_json(os.path.join(output_dir, "metrics.json"), {"history": history, "metrics": metrics})

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"FDA validation accuracy: {fda_val_acc * 100:.2f}%")
    print(f"Target test accuracy: {target_test_acc * 100:.2f}%")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
