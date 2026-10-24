import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from model import ArithmeticCNN, CLASSES, CLASS_TO_IDX

ARTIFACTS = Path("artifacts")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

class SyntheticOperatorDataset(Dataset):
    """Generate simple rendered +, -, x symbols. Real handwritten samples are still recommended."""
    def __init__(self, count, transform=None, seed=100):
        self.count = count
        self.transform = transform
        self.rng = random.Random(seed)
        self.font_path = find_font()

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # Use deterministic per-index RNG so examples are reproducible.
        rng = random.Random(idx + 982451653)
        label = rng.choice(["+", "-", "x", "/"])
        img = Image.new("L", (56, 56), color=255)
        draw = ImageDraw.Draw(img)
        size = rng.randint(25, 42)
        try:
            font = ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x0 = (56 - w) // 2 - bbox[0] + rng.randint(-3, 3)
        y0 = (56 - h) // 2 - bbox[1] + rng.randint(-3, 3)
        draw.text((x0, y0), label, fill=rng.randint(0, 90), font=font)
        angle = rng.uniform(-18, 18)
        img = img.rotate(angle, fillcolor=255)
        if rng.random() < 0.25:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0, 0.7)))
        # Invert to white strokes on black background like MNIST.
        img = Image.eval(img, lambda p: 255 - p)
        tensor = transforms.ToTensor()(img)
        # Resize and pad into the same 28x28 input convention.
        tensor = transforms.Resize((28, 28), antialias=True)(tensor)
        # Slight translations/rotations for operators.
        return tensor, CLASS_TO_IDX[label]

class RemappedMNIST(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # torchvision MNIST transform already yields a 1x28x28 tensor, black bg/white strokes.
        return x, int(y)

def make_loaders(batch_size, limit=None):
    base_transform = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.10, 0.10), scale=(0.88, 1.12), fill=0),
        transforms.ToTensor(),
    ])
    train_mnist = datasets.MNIST("data", train=True, download=True, transform=base_transform)
    test_mnist = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    if limit:
        train_mnist = torch.utils.data.Subset(train_mnist, range(min(limit, len(train_mnist))))
        test_mnist = torch.utils.data.Subset(test_mnist, range(min(max(1000, limit // 5), len(test_mnist))))
    # Add synthetic operators to the training set. Their labels are 10,11,12.
    operator_ds = SyntheticOperatorDataset(count=max(12000, len(train_mnist) // 5))
    combined = ConcatDataset([RemappedMNIST(train_mnist), operator_ds])
    train_loader = DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(RemappedMNIST(test_mnist), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    all_y, all_pred = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        count += x.size(0)
        all_y.extend(y.detach().cpu().tolist())
        all_pred.extend(pred.detach().cpu().tolist())
    return total_loss / max(count, 1), correct / max(count, 1), all_y, all_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Small subset for a quick smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    ARTIFACTS.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")
    train_loader, test_loader = make_loaders(args.batch_size, args.limit)
    model = ArithmeticCNN().to(DEVICE)  # random initialization: no pretrained weights
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc, y_true, y_pred = run_epoch(model, test_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | digit test loss {te_loss:.4f} acc {te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES}, ARTIFACTS / "arithmetic_cnn.pt")

    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["test_loss"], label="digit test")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["test_acc"], label="digit test")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(ARTIFACTS / "training_curves.png", dpi=160); plt.close(fig)

    # Final digit test metrics; operator classes are not represented in MNIST test data.
    _, _, y_true, y_pred = run_epoch(model, test_loader, criterion)
    labels = list(range(10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES[:10])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    fig.tight_layout(); fig.savefig(ARTIFACTS / "digit_confusion_matrix.png", dpi=160); plt.close(fig)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES[:10], zero_division=0)
    (ARTIFACTS / "digit_classification_report.txt").write_text(report)
    print("\nDigit-only test report:\n", report)
    print(f"\nSaved model and plots in {ARTIFACTS.resolve()}")
    print("Note: this test accuracy measures MNIST digits only, not complete arithmetic expressions or operators.")

if __name__ == "__main__":
    main()
