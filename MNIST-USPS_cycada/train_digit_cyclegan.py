import argparse
import os
import time

import torch
import torch.nn as nn

from digit_uda.common import (
    ForeverDataIterator,
    create_writer,
    ensure_dir,
    load_checkpoint,
    log_scalars,
    plot_history,
    save_checkpoint,
    save_json,
    set_seed,
)
from digit_uda.cyclegan import (
    LeastSquaresGanLoss,
    PatchDiscriminator,
    ResNetGenerator,
    build_unpaired_digit_loaders,
    load_frozen_classifier,
    low_frequency_fft_loss,
    save_translation_preview,
    semantic_consistency_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train digit CycleGAN variants for MNIST -> USPS.")
    parser.add_argument("--source", type=str, default="mnist", choices=["mnist", "usps"])
    parser.add_argument("--target", type=str, default="usps", choices=["mnist", "usps"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/cyclegan/mnist_to_usps")
    parser.add_argument(
        "--variant",
        type=str,
        default="freq",
        choices=["vanilla", "freq", "sem", "freq_sem"],
        help="CycleGAN variant. Use `freq` or `sem` for the project experiments.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--lambda-freq", type=float, default=1.0)
    parser.add_argument("--lambda-sem", type=float, default=1.0)
    parser.add_argument("--low-freq-ratio", type=float, default=0.25)
    parser.add_argument("--semantic-checkpoint", type=str, default="")
    parser.add_argument("--teacher-image-size", type=int, default=224)
    parser.add_argument("--preview-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def resolve_project_path(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(project_root, path))


def resolve_data_root(data_root: str, project_root: str, source: str, target: str) -> str:
    requested_root = resolve_project_path(data_root, project_root)
    project_data_root = os.path.join(project_root, "data")
    requested_ok = os.path.isdir(os.path.join(requested_root, source)) or os.path.isdir(
        os.path.join(requested_root, target)
    )
    project_ok = os.path.isdir(os.path.join(project_data_root, source)) or os.path.isdir(
        os.path.join(project_data_root, target)
    )
    if requested_ok:
        return requested_root
    if project_ok:
        print(f"Using project-local data root instead of missing dataset path: {project_data_root}")
        return project_data_root
    return requested_root


def resolve_semantic_checkpoint(path: str, project_root: str) -> str:
    requested_path = resolve_project_path(path, project_root)
    if os.path.isfile(requested_path):
        return requested_path

    fallback = os.path.join(project_root, "outputs", "source_only", "mnist_to_usps", "best.pth")
    if os.path.isfile(fallback):
        print(f"Using fallback semantic checkpoint: {fallback}")
        return fallback

    raise FileNotFoundError(
        "Semantic checkpoint not found. Please provide --semantic-checkpoint or train source-only first."
    )


def main() -> None:
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_data_root(args.data_root, project_root, args.source, args.target)
    args.output_dir = resolve_project_path(args.output_dir, project_root)
    if args.resume:
        args.resume = resolve_project_path(args.resume, project_root)
    if "sem" in args.variant:
        args.semantic_checkpoint = resolve_semantic_checkpoint(args.semantic_checkpoint, project_root)

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = create_writer(args.output_dir)

    print(f"Using device: {device}")
    print(f"Variant: {args.variant}")
    print(f"Source: {args.source} | Target: {args.target}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")

    loaders = build_unpaired_digit_loaders(
        source=args.source,
        target=args.target,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    source_loader = loaders["source_train"]
    target_loader = loaders["target_train"]
    steps_per_epoch = args.steps_per_epoch or max(len(source_loader), len(target_loader))

    generator_a2b = ResNetGenerator().to(device)
    generator_b2a = ResNetGenerator().to(device)
    discriminator_a = PatchDiscriminator().to(device)
    discriminator_b = PatchDiscriminator().to(device)

    criterion_gan = LeastSquaresGanLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    teacher = None
    if "sem" in args.variant:
        teacher = load_frozen_classifier(args.semantic_checkpoint, device=device)

    optimizer_g = torch.optim.Adam(
        list(generator_a2b.parameters()) + list(generator_b2a.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    optimizer_d_a = torch.optim.Adam(discriminator_a.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d_b = torch.optim.Adam(discriminator_b.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs)
    scheduler_d_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d_a, T_max=args.epochs)
    scheduler_d_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d_b, T_max=args.epochs)

    best_path = os.path.join(args.output_dir, "best.pth")
    last_path = os.path.join(args.output_dir, "last.pth")
    preview_dir = os.path.join(args.output_dir, "preview")
    history = []
    best_generator_loss = float("inf")
    start_epoch = 1

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            modules={
                "G_A2B": generator_a2b,
                "G_B2A": generator_b2a,
                "D_A": discriminator_a,
                "D_B": discriminator_b,
            },
            optimizers={
                "optimizer_g": optimizer_g,
                "optimizer_d_a": optimizer_d_a,
                "optimizer_d_b": optimizer_d_b,
            },
            schedulers={
                "scheduler_g": scheduler_g,
                "scheduler_d_a": scheduler_d_a,
                "scheduler_d_b": scheduler_d_b,
            },
            map_location=device,
        )
        state = checkpoint.get("state", {})
        history = state.get("history", [])
        best_generator_loss = state.get("best_generator_loss", best_generator_loss)
        start_epoch = state.get("epoch", 0) + 1
        print(f"Resumed from: {args.resume} (next epoch: {start_epoch})")

    fixed_source_images, _ = next(iter(source_loader))
    fixed_target_images, _ = next(iter(target_loader))
    fixed_source_images = fixed_source_images.to(device)
    fixed_target_images = fixed_target_images.to(device)

    source_iter = ForeverDataIterator(source_loader)
    target_iter = ForeverDataIterator(target_loader)
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        generator_a2b.train()
        generator_b2a.train()
        discriminator_a.train()
        discriminator_b.train()

        epoch_log = {
            "epoch": epoch,
            "generator_loss": 0.0,
            "discriminator_a_loss": 0.0,
            "discriminator_b_loss": 0.0,
            "adv_loss": 0.0,
            "cycle_loss": 0.0,
            "identity_loss": 0.0,
            "freq_loss": 0.0,
            "sem_loss": 0.0,
        }

        for _ in range(steps_per_epoch):
            real_a, labels_a = next(source_iter)
            real_b, _ = next(target_iter)
            real_a = real_a.to(device, non_blocking=True)
            labels_a = labels_a.to(device, non_blocking=True)
            real_b = real_b.to(device, non_blocking=True)

            optimizer_g.zero_grad()

            identity_a = generator_b2a(real_a)
            identity_b = generator_a2b(real_b)
            loss_identity = (
                criterion_identity(identity_a, real_a) + criterion_identity(identity_b, real_b)
            ) * args.lambda_identity

            fake_b = generator_a2b(real_a)
            fake_a = generator_b2a(real_b)
            cycle_a = generator_b2a(fake_b)
            cycle_b = generator_a2b(fake_a)

            loss_adv = criterion_gan(discriminator_b(fake_b), True) + criterion_gan(discriminator_a(fake_a), True)
            loss_cycle = (
                criterion_cycle(cycle_a, real_a) + criterion_cycle(cycle_b, real_b)
            ) * args.lambda_cycle

            freq_loss_value = real_a.new_tensor(0.0)
            if "freq" in args.variant:
                freq_loss_value = (
                    low_frequency_fft_loss(real_a, cycle_a, ratio=args.low_freq_ratio)
                    + low_frequency_fft_loss(real_b, cycle_b, ratio=args.low_freq_ratio)
                ) * args.lambda_freq

            sem_loss_value = real_a.new_tensor(0.0)
            if teacher is not None:
                sem_loss_value = semantic_consistency_loss(
                    teacher=teacher,
                    fake_target=fake_b,
                    labels=labels_a,
                    image_size=args.teacher_image_size,
                ) * args.lambda_sem

            generator_loss = loss_adv + loss_cycle + loss_identity + freq_loss_value + sem_loss_value
            generator_loss.backward()
            optimizer_g.step()

            optimizer_d_a.zero_grad()
            loss_d_a_real = criterion_gan(discriminator_a(real_a), True)
            loss_d_a_fake = criterion_gan(discriminator_a(fake_a.detach()), False)
            loss_d_a = 0.5 * (loss_d_a_real + loss_d_a_fake)
            loss_d_a.backward()
            optimizer_d_a.step()

            optimizer_d_b.zero_grad()
            loss_d_b_real = criterion_gan(discriminator_b(real_b), True)
            loss_d_b_fake = criterion_gan(discriminator_b(fake_b.detach()), False)
            loss_d_b = 0.5 * (loss_d_b_real + loss_d_b_fake)
            loss_d_b.backward()
            optimizer_d_b.step()

            epoch_log["generator_loss"] += generator_loss.item()
            epoch_log["discriminator_a_loss"] += loss_d_a.item()
            epoch_log["discriminator_b_loss"] += loss_d_b.item()
            epoch_log["adv_loss"] += loss_adv.item()
            epoch_log["cycle_loss"] += loss_cycle.item()
            epoch_log["identity_loss"] += loss_identity.item()
            epoch_log["freq_loss"] += freq_loss_value.item()
            epoch_log["sem_loss"] += sem_loss_value.item()

        for key in list(epoch_log.keys()):
            if key != "epoch":
                epoch_log[key] /= steps_per_epoch

        epoch_log["lr_g"] = optimizer_g.param_groups[0]["lr"]
        epoch_log["lr_d_a"] = optimizer_d_a.param_groups[0]["lr"]
        epoch_log["lr_d_b"] = optimizer_d_b.param_groups[0]["lr"]
        history.append(epoch_log)
        log_scalars(writer, epoch_log, epoch)

        scheduler_g.step()
        scheduler_d_a.step()
        scheduler_d_b.step()

        if args.preview_every > 0 and (epoch % args.preview_every == 0 or epoch == args.epochs):
            preview_path = save_translation_preview(
                output_dir=preview_dir,
                epoch=epoch,
                generator_a2b=generator_a2b,
                generator_b2a=generator_b2a,
                real_a=fixed_source_images,
                real_b=fixed_target_images,
            )
            print(f"Saved preview: {preview_path}")

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"G={epoch_log['generator_loss']:.4f} "
            f"D_A={epoch_log['discriminator_a_loss']:.4f} "
            f"D_B={epoch_log['discriminator_b_loss']:.4f} "
            f"adv={epoch_log['adv_loss']:.4f} "
            f"cycle={epoch_log['cycle_loss']:.4f} "
            f"id={epoch_log['identity_loss']:.4f} "
            f"freq={epoch_log['freq_loss']:.4f} "
            f"sem={epoch_log['sem_loss']:.4f}"
        )

        state = {
            "epoch": epoch,
            "history": history,
            "best_generator_loss": best_generator_loss,
            "args": vars(args),
            "elapsed_seconds": time.time() - train_start,
        }
        save_checkpoint(
            last_path,
            modules={
                "G_A2B": generator_a2b,
                "G_B2A": generator_b2a,
                "D_A": discriminator_a,
                "D_B": discriminator_b,
            },
            optimizers={
                "optimizer_g": optimizer_g,
                "optimizer_d_a": optimizer_d_a,
                "optimizer_d_b": optimizer_d_b,
            },
            schedulers={
                "scheduler_g": scheduler_g,
                "scheduler_d_a": scheduler_d_a,
                "scheduler_d_b": scheduler_d_b,
            },
            state=state,
        )
        if epoch_log["generator_loss"] < best_generator_loss:
            best_generator_loss = epoch_log["generator_loss"]
            state["best_generator_loss"] = best_generator_loss
            save_checkpoint(
                best_path,
                modules={
                    "G_A2B": generator_a2b,
                    "G_B2A": generator_b2a,
                    "D_A": discriminator_a,
                    "D_B": discriminator_b,
                },
                optimizers={
                    "optimizer_g": optimizer_g,
                    "optimizer_d_a": optimizer_d_a,
                    "optimizer_d_b": optimizer_d_b,
                },
                schedulers={
                    "scheduler_g": scheduler_g,
                    "scheduler_d_a": scheduler_d_a,
                    "scheduler_d_b": scheduler_d_b,
                },
                state=state,
            )

    metrics = {
        "source": args.source,
        "target": args.target,
        "variant": args.variant,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "lambda_cycle": args.lambda_cycle,
        "lambda_identity": args.lambda_identity,
        "lambda_freq": args.lambda_freq,
        "lambda_sem": args.lambda_sem,
        "low_freq_ratio": args.low_freq_ratio,
        "semantic_checkpoint": args.semantic_checkpoint,
        "best_generator_loss": best_generator_loss,
        "selection_checkpoint": best_path if os.path.exists(best_path) else last_path,
    }
    save_json(os.path.join(args.output_dir, "metrics.json"), {"history": history, "metrics": metrics})
    plot_history(
        history,
        args.output_dir,
        groups={
            "loss": [
                "generator_loss",
                "discriminator_a_loss",
                "discriminator_b_loss",
                "adv_loss",
                "cycle_loss",
                "identity_loss",
                "freq_loss",
                "sem_loss",
            ],
            "learning_rate": ["lr_g", "lr_d_a", "lr_d_b"],
        },
    )

    if writer is not None:
        writer.close()

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
