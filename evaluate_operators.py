import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import ArithmeticCNN, CLASSES, CLASS_TO_IDX

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
OPERATORS = ["+", "-", "x", "/"]


class OperatorTestDataset(Dataset):
    """Fresh, balanced synthetic operator test samples."""

    def __init__(self, samples_per_class=1000):
        self.samples_per_class = samples_per_class
        self.labels = [
            op for op in OPERATORS
            for _ in range(samples_per_class)
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Separate deterministic sequence from training sample indices.
        rng = random.Random(800000 + idx)
        label = self.labels[idx]

        img = Image.new("L", (56, 56), color=255)
        draw = ImageDraw.Draw(img)

        size = rng.randint(25, 42)
        try:
            font = (
                ImageFont.truetype(FONT_PATH, size)
                if FONT_PATH else ImageFont.load_default()
            )
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        x0 = (56 - w) // 2 - bbox[0] + rng.randint(-3, 3)
        y0 = (56 - h) // 2 - bbox[1] + rng.randint(-3, 3)

        draw.text(
            (x0, y0), label,
            fill=rng.randint(0, 90),
            font=font
        )

        img = img.rotate(
            rng.uniform(-18, 18),
            fillcolor=255
        )

        if rng.random() < 0.25:
            img = img.filter(
                ImageFilter.GaussianBlur(
                    radius=rng.uniform(0.1, 0.7)
                )
            )

        # Match the training pipeline: white strokes, black background.
        img = Image.eval(img, lambda p: 255 - p)
        x = transforms.ToTensor()(img)
        x = transforms.Resize((28, 28), antialias=True)(x)

        return x, CLASS_TO_IDX[label]


def main():
    checkpoint_path = ARTIFACTS / "arithmetic_cnn.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    # Ensure the checkpoint and current class mapping agree.
    if checkpoint.get("classes") != CLASSES:
        raise ValueError(
            "Checkpoint class order differs from model.py CLASSES. "
            "Check the class mapping before evaluating."
        )

    model = ArithmeticCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = OperatorTestDataset(samples_per_class=1000)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=0
    )

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            predictions = logits.argmax(dim=1).cpu()

            y_true.extend(y.tolist())
            y_pred.extend(predictions.tolist())

    operator_ids = [CLASS_TO_IDX[op] for op in OPERATORS]

    print(f"Device: {DEVICE}")
    print("Synthetic operator test report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=operator_ids,
            target_names=OPERATORS,
            digits=4,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=operator_ids)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=OPERATORS
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Synthetic Operator Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "operator_confusion_matrix.png", dpi=160)
    plt.close()

    report = classification_report(
        y_true,
        y_pred,
        labels=operator_ids,
        target_names=OPERATORS,
        digits=4,
        zero_division=0
    )
    (ARTIFACTS / "operator_classification_report.txt").write_text(report)

    print("Saved operator report and confusion matrix in artifacts/")


if __name__ == "__main__":
    main()
