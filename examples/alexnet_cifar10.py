"""Train a small AlexNet-style classifier while trainctl supervises the process.

The script intentionally prints metrics as simple key/value pairs so trainctl's
GenericTrainingAdapter can parse them without any trainctl-specific Python API.

Use --dataset fake for a download-free smoke test or --dataset cifar10 for a
real CIFAR-10 run.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AlexNetCIFAR(nn.Module):
    """AlexNet-style network adapted to 32x32 CIFAR images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fake", "cifar10"), default="fake")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./outputs/alexnet"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--fail-at-step",
        type=int,
        default=None,
        help="Intentionally raise an error at this step to test RunFailed notifications.",
    )
    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if args.dataset == "cifar10":
        return datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    size = max(args.steps * args.batch_size, args.batch_size)
    return datasets.FakeData(
        size=size,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=transform,
    )


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_dir / f"checkpoint-{step}.pt",
    )


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")

    random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = AlexNetCIFAR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(
        f"device={device} dataset={args.dataset} total_steps={args.steps}",
        flush=True,
    )

    batches = itertools.cycle(loader)
    model.train()
    for step in range(1, args.steps + 1):
        images, labels = next(batches)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if step % max(args.log_every, 1) == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step} loss={loss.item():.6f} learning_rate={lr:.8g}",
                flush=True,
            )

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(args.checkpoint_dir, step, model, optimizer)

        if args.fail_at_step is not None and step == args.fail_at_step:
            raise RuntimeError(
                f"Intentional failure at step {step}; this is a trainctl notification demo."
            )

    if not args.checkpoint_every or args.steps % args.checkpoint_every != 0:
        save_checkpoint(args.checkpoint_dir, args.steps, model, optimizer)

    print(f"training_complete=true step={args.steps}", flush=True)


if __name__ == "__main__":
    main()
