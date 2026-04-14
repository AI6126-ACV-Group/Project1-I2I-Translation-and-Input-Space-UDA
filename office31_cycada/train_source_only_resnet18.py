import argparse
import os
import time

import torch
import torch.nn as nn

from office31_uda.common import (
    NUM_CLASSES,
    ResNet18FeatureNet,
    build_source_only_loaders,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train source-only ResNet-18 baseline for Office-31 Amazon -> Webcam."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/source_only/amazon_to_webcam")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    setup_logging(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = create_writer(args.output_dir)
    print(f"Using device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Split dir: {args.split_dir}")
    print(f"Output dir: {args.output_dir}")

    loaders = build_source_only_loaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
    )

    model = ResNet18FeatureNet(num_classes=NUM_CLASSES, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_path = os.path.join(args.output_dir, "best.pth")
    last_path = os.path.join(args.output_dir, "last.pth")
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
        train_start = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=loaders["src_train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            val_loss, val_acc = evaluate_classifier(
                model=model,
                loader=loaders["src_val"],
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
                "elapsed_seconds": time.time() - train_start,
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
                args.output_dir,
                history=history,
                summary={"best_val_acc": best_val_acc, "latest_epoch": epoch},
            )
            plot_history(
                history,
                args.output_dir,
                groups={
                    "loss": ["train_loss", "val_loss"],
                    "accuracy": ["train_acc", "val_acc"],
                    "learning_rate": ["lr"],
                },
            )

    eval_path = best_path if os.path.exists(best_path) else last_path
    checkpoint = load_checkpoint(eval_path, modules={"model": model}, map_location=device)
    best_val_acc = checkpoint.get("state", {}).get("best_val_acc", best_val_acc)
    history = checkpoint.get("state", {}).get("history", history)

    src_val_loss, src_val_acc = evaluate_classifier(
        model=model,
        loader=loaders["src_val"],
        criterion=criterion,
        device=device,
    )
    tgt_test_loss, tgt_test_acc = evaluate_classifier(
        model=model,
        loader=loaders["tgt_test"],
        criterion=criterion,
        device=device,
    )

    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "pretrained": args.pretrained,
        "best_val_acc": best_val_acc,
        "source_val_loss": src_val_loss,
        "source_val_acc": src_val_acc,
        "target_test_loss": tgt_test_loss,
        "target_test_acc": tgt_test_acc,
        "selection_checkpoint": eval_path,
        "metric": "top1_accuracy",
    }

    save_json(os.path.join(args.output_dir, "metrics.json"), {"history": history, "metrics": metrics})

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Source validation accuracy: {src_val_acc * 100:.2f}%")
    print(f"Target test accuracy: {tgt_test_acc * 100:.2f}%")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
