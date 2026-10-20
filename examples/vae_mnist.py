"""Train a small MNIST VAE while trainctl supervises the process.

Use --dataset fake for a fast, download-free smoke test. The training loop emits
"step=... loss=..." lines that trainctl can parse from ordinary stdout.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.flatten(1)
        mu, logvar = self.encode(flat)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fake", "mnist"), default="fake")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./outputs/vae"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=20)
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
    transform = transforms.ToTensor()
    if args.dataset == "mnist":
        return datasets.MNIST(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    size = max(args.steps * args.batch_size, args.batch_size)
    return datasets.FakeData(
        size=size,
        image_size=(1, 28, 28),
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

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"device={device} dataset={args.dataset} total_steps={args.steps}",
        flush=True,
    )

    batches = itertools.cycle(loader)
    model.train()
    for step in range(1, args.steps + 1):
        images, _ = next(batches)
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar = model(images)

        target = images.flatten(1)
        reconstruction_loss = F.binary_cross_entropy(
            reconstruction, target, reduction="sum"
        ) / images.shape[0]
        kl_loss = (
            -0.5
            * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            / images.shape[0]
        )
        loss = reconstruction_loss + kl_loss
        loss.backward()
        optimizer.step()

        if step % max(args.log_every, 1) == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"loss={loss.item():.6f}",
                        f"reconstruction_loss={reconstruction_loss.item():.6f}",
                        f"kl_loss={kl_loss.item():.6f}",
                        f"learning_rate={lr:.8g}",
                    ]
                ),
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
