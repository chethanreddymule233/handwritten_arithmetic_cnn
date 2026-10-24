from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageOps
import cv2

from model import ArithmeticCNN, CLASSES, CLASS_TO_IDX
from train_real_operators import make_datasets, DEVICE

SEED = 42
ORIGINAL = Path("artifacts/arithmetic_cnn.pt")
OUTPUT = Path("artifacts/arithmetic_cnn_handwritten_x.pt")
HANDWRITTEN_DIR = Path("data/my_handwritten/x")

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_handwritten_image(path):
    """Convert a dark-ink paper crop into the app's 28x28 input format."""
    image = Image.open(path).convert("L")
    gray = np.array(image)

    # Paper/background -> black; dark handwritten strokes -> white.
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 11
    )

    # Trim background around the strokes, as app.py does.
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        raise ValueError(f"No strokes found in {path}")

    crop = binary[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (nw, nh), interpolation=interpolation)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - nw) // 2, (28 - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized

    return Image.fromarray(canvas)


class HandwrittenXDataset(Dataset):
    def __init__(self, folder, augment=False):
        self.files = sorted(folder.glob("*.png"))
        self.augment = augment

        if not self.files:
            raise RuntimeError(f"No PNG images found in {folder}")

        # Convert each paper crop once. Apply small variations during training.
        self.images = [prepare_handwritten_image(p) for p in self.files]
        self.augmentation = transforms.RandomAffine(
            degrees=7, translate=(0.05, 0.05), fill=0
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            image = self.augmentation(image)

        return self.to_tensor(image), CLASS_TO_IDX["x"]


def evaluate(model, loader):
    model.eval()
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            count += labels.numel()

    return correct / max(count, 1)


def main():
    seed_everything(SEED)

    if not ORIGINAL.exists():
        raise FileNotFoundError(f"Original checkpoint missing: {ORIGINAL}")

    # Existing loaders preserve the original MNIST, synthetic, and real
    # operator training data and the original validation split.
    base_loader, digit_test_loader, operator_val_loader = make_datasets()

    base_dataset = base_loader.dataset
    handwritten = HandwrittenXDataset(HANDWRITTEN_DIR, augment=True)

    # Sample 10% handwritten-X examples per epoch, approximately.
    # This adds emphasis without replacing the existing dataset.
    base_size = len(base_dataset)
    x_weight = base_size * 0.10 / (0.90 * len(handwritten))
    weights = [1.0] * base_size + [x_weight] * len(handwritten)

    combined = ConcatDataset([base_dataset, handwritten])
    sampler = WeightedRandomSampler(
        weights, num_samples=base_size, replacement=True
    )
    train_loader = DataLoader(
        combined, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )

    checkpoint = torch.load(ORIGINAL, map_location=DEVICE)
    model = ArithmeticCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )

    print("Device:", DEVICE)
    print("Original checkpoint:", ORIGINAL)
    print("Handwritten X samples:", len(handwritten))
    print("Existing training samples:", base_size)
    print("Saving experimental model to:", OUTPUT)

    baseline_digit = evaluate(model, digit_test_loader)
    baseline_operator = evaluate(model, operator_val_loader)
    print(
        f"Baseline | digit accuracy: {baseline_digit:.4%} | "
        f"operator validation accuracy: {baseline_operator:.4%}"
    )

    best_score = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

        digit_acc = evaluate(model, digit_test_loader)
        operator_acc = evaluate(model, operator_val_loader)
        score = (digit_acc + operator_acc) / 2

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss {total_loss / max(total_count, 1):.4f} | "
            f"digit accuracy {digit_acc:.4%} | "
            f"operator validation accuracy {operator_acc:.4%}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "fine_tuned_from": str(ORIGINAL),
                    "handwritten_x_samples": len(handwritten),
                },
                OUTPUT,
            )
            print("Saved best experimental checkpoint.")

    print("\nFine-tuning finished.")
    print("Original model remains unchanged:", ORIGINAL)
    print("Experimental model:", OUTPUT)


if __name__ == "__main__":
    main()
