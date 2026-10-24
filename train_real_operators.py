import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image, ImageOps
from sklearn.metrics import classification_report

from model import ArithmeticCNN, CLASSES, CLASS_TO_IDX
from train import SyntheticOperatorDataset, RemappedMNIST, seed_everything

ARTIFACTS = Path("artifacts")
SOURCE = Path("bhmsds/symbols")
TEST_ROOT = Path("handwritten_test")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

PREFIX_TO_LABEL = {
    "plus": "+",
    "minus": "-",
    "x": "x",
    "slash": "/",
}


class RealOperatorDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        prefix = path.name.split("-")[0]
        label = PREFIX_TO_LABEL[prefix]

        # Dataset images have dark ink on a white background.
        # Convert to white ink on black, matching MNIST.
        image = Image.open(path).convert("L")
        image = ImageOps.invert(image)

        if self.transform:
            image = self.transform(image)

        return image, CLASS_TO_IDX[label]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_datasets(seed=42):
    rng = random.Random(seed)

    # Exclude the 400 held-out images from the training source.
    heldout_hashes = set()
    for folder in PREFIX_TO_LABEL.keys():
        folder_path = TEST_ROOT / folder
        for path in folder_path.iterdir():
            if path.is_file() and path.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".bmp"
            }:
                heldout_hashes.add(file_hash(path))

    train_files = []
    val_files = []

    for prefix in PREFIX_TO_LABEL:
        files = [
            path for path in SOURCE.glob(f"{prefix}-*.png")
            if file_hash(path) not in heldout_hashes
        ]
        rng.shuffle(files)

        if len(files) < 10:
            raise RuntimeError(
                f"Not enough source images for {prefix}: {len(files)}"
            )

        split = int(len(files) * 0.85)
        train_files.extend(files[:split])
        val_files.extend(files[split:])

        print(
            f"{prefix}: {len(files)} available, "
            f"{split} train, {len(files) - split} validation"
        )

    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=7,
            translate=(0.06, 0.06),
            fill=0
        ),
        transforms.ToTensor(),
    ])

    val_transform = transforms.ToTensor()

    real_train = RealOperatorDataset(train_files, train_transform)
    real_val = RealOperatorDataset(val_files, val_transform)

    mnist_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=12,
            translate=(0.10, 0.10),
            scale=(0.88, 1.12),
            fill=0
        ),
        transforms.ToTensor(),
    ])

    mnist_train = datasets.MNIST(
        "data", train=True, download=True, transform=mnist_transform
    )
    mnist_test = datasets.MNIST(
        "data", train=False, download=True,
        transform=transforms.ToTensor()
    )

    # Keep a modest synthetic set alongside real handwritten operators.
    synthetic = SyntheticOperatorDataset(count=4000, seed=seed + 100)

    combined = ConcatDataset([
        RemappedMNIST(mnist_train),
        synthetic,
        real_train,
    ])

    return (
        DataLoader(combined, batch_size=128, shuffle=True, num_workers=0),
        DataLoader(
            RemappedMNIST(mnist_test),
            batch_size=256, shuffle=False, num_workers=0
        ),
        DataLoader(real_val, batch_size=128, shuffle=False, num_workers=0),
    )


def evaluate(model, loader, criterion=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)

            if criterion is not None:
                total_loss += criterion(logits, y).item() * x.size(0)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            count += x.size(0)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    return (
        total_loss / max(count, 1),
        correct / max(count, 1),
        all_true,
        all_pred,
    )


def main():
    seed_everything(42)
    ARTIFACTS.mkdir(exist_ok=True)

    print("Device:", DEVICE)
    train_loader, digit_loader, operator_val_loader = make_datasets()

    # Fresh random initialization: no pretrained weights.
    model = ArithmeticCNN(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )

    best_score = -1.0
    epochs = 8

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_count += x.size(0)

        _, digit_acc, _, _ = evaluate(model, digit_loader)
        _, op_acc, op_true, op_pred = evaluate(
            model, operator_val_loader
        )

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"digit acc {digit_acc:.4f} | "
            f"operator validation acc {op_acc:.4f}"
        )

        # Balance digit recognition and operator generalization.
        score = (digit_acc + op_acc) / 2

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                },
                ARTIFACTS / "arithmetic_cnn.pt",
            )
            print("  Saved best model.")

    print("\nOperator validation report:")
    print(classification_report(
        op_true,
        op_pred,
        labels=[CLASS_TO_IDX[s] for s in ["+", "-", "x", "/"]],
        target_names=["+", "-", "x", "/"],
        zero_division=0,
        digits=4,
    ))

    print("Training complete.")
    print("Checkpoint:", ARTIFACTS / "arithmetic_cnn.pt")
    print("The 400 handwritten_test images were excluded from training.")


if __name__ == "__main__":
    main()
