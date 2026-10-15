"""Baseline: a fine-tuned ImageNet CNN on raw image patches -> SMALL / BIG.

Augmentation is limited to flips, 90-degree rotations and colour jitter: no scaling or
cropping, which would destroy the size signal. PyTorch is used only here, never at
inference time.

    pip install torch torchvision          # CPU is fine; a GPU is faster
    python -m baselines.cnn_patch [--arch mobilenet_v3_small|resnet18] [--epochs 15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.common import leave_one_item_out, load_patches, write_report
from src.config import DataPaths


def make_fit_predict(arch: str, epochs: int, pretrained: bool, seed: int = 0):
    """Build the CNN baseline's training-and-prediction function.

    PyTorch is imported here, so the rest of the project never needs it.

    Args:
        arch: "mobilenet_v3_small" or "resnet18".
        epochs: Training epochs per fold.
        pretrained: Start from ImageNet weights (downloaded on first use).
        seed: Random seed, for repeatable folds.

    Returns:
        The fit-and-predict function and the device it runs on ("cuda" or "cpu").
    """
    import torch
    import torchvision

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def to_tensor(patches: np.ndarray) -> torch.Tensor:
        """Convert BGR patches into a normalised RGB tensor.

        Args:
            patches: Patches of shape `(N, H, W, 3)`.

        Returns:
            A float tensor of shape `(N, 3, H, W)`, normalised with ImageNet statistics.
        """
        x = torch.from_numpy(patches[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255  # BGR -> RGB
        return (x - mean) / std

    def build():
        """Create the network with a fresh two-class output layer.

        Returns:
            The model, on the chosen device.
        """
        if arch == "resnet18":
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            m.fc = torch.nn.Linear(int(m.fc.in_features), 2)
        else:
            m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
            last = m.classifier[-1]
            assert isinstance(last, torch.nn.Linear)
            m.classifier[-1] = torch.nn.Linear(last.in_features, 2)
        return m.to(device)

    def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        """Randomly flip, rotate by 90° steps and jitter brightness and contrast.

        There is no scaling or cropping, which would destroy the size signal.

        Args:
            x: A batch of normalised patches.
            g: Random generator, for repeatable augmentation.

        Returns:
            The augmented batch.
        """
        if torch.rand(1, generator=g) < 0.5:
            x = x.flip(3)
        if torch.rand(1, generator=g) < 0.5:
            x = x.flip(2)
        x = torch.rot90(x, int(torch.randint(0, 4, (1,), generator=g)), dims=(2, 3))
        # colour jitter: per-image brightness / contrast in normalised space
        b = (torch.rand(x.shape[0], 1, 1, 1, generator=g) - 0.5) * 0.4
        c = 1 + (torch.rand(x.shape[0], 1, 1, 1, generator=g) - 0.5) * 0.4
        return x * c + b

    def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """Fine-tune a fresh network on one fold and predict the held-out patches.

        Args:
            X_train: Training patches.
            y_train: Training labels (True = BIG).
            X_test: Patches to classify.

        Returns:
            Predicted BIG flags for `X_test`.
        """
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        model = build()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        xt, yt = to_tensor(X_train), torch.from_numpy(y_train.astype(np.int64))
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(xt), generator=g)
            for i in range(0, len(xt), 16):
                idx = perm[i : i + 16]
                xb = augment(xt[idx], g).to(device)
                loss = torch.nn.functional.cross_entropy(model(xb), yt[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            return model(to_tensor(X_test).to(device)).argmax(1).cpu().numpy().astype(bool)

    return fit_predict, device


def main() -> None:
    """Command-line entry point: run the CNN baseline."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DataPaths().root)
    p.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--px", type=int, default=224)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    try:
        fit_predict, device = make_fit_predict(args.arch, args.epochs, not args.no_pretrained)
    except ImportError:
        raise SystemExit("PyTorch is not installed: pip install torch torchvision") from None

    dp = DataPaths(args.data)
    rows, patches, meta = load_patches(dp, args.px)
    preds = leave_one_item_out(rows, patches, meta["thresholds"], fit_predict)
    rep = write_report(
        "cnn",
        preds,
        {"arch": args.arch, "epochs": args.epochs, "patch_px": args.px, "side_mm": meta["side_mm"], "device": device},
        dp.synthetic_flag.exists(),
    )
    print(f"CNN ({args.arch}) Small/Big (leave-one-item-out): accuracy {rep['overall']['accuracy']:.3f}, F1 {rep['overall']['f1_big']:.3f}")
    for c, v in rep["per_crop"].items():
        print(f"  {c:11s} accuracy {v['accuracy']:.3f}  F1 {v['f1_big']:.3f}")


if __name__ == "__main__":
    main()
