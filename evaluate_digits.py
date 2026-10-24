
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from model import ArithmeticCNN, CLASSES


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CHECKPOINT = Path("artifacts/arithmetic_cnn.pt")
DATA_DIR = Path("data")
BATCH_SIZE = 256

DIGITS = [str(i) for i in range(10)]

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )


# --------------------------------------------------
# 2. Select device
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 3. Load checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

if "classes" in checkpoint and checkpoint["classes"] != CLASSES:
    raise ValueError(
        f"Class mismatch!\n"
        f"Checkpoint: {checkpoint['classes']}\n"
        f"Current: {CLASSES}"
    )

model = ArithmeticCNN(num_classes=len(CLASSES))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Loaded checkpoint:", CHECKPOINT)


# --------------------------------------------------
# 4. Load MNIST test set
# --------------------------------------------------

# MNIST images are already white strokes on a black
# background. Do not invert them.
test_transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("MNIST test images:", len(test_dataset))


# --------------------------------------------------
# 5. Run predictions
# --------------------------------------------------

true_labels = []
predicted_labels = []

correct_digits = 0
total_digits = 0
predicted_as_operator = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())

        correct_digits += (
            predictions == labels
        ).sum().item()

        total_digits += labels.size(0)

        predicted_as_operator += (
            predictions >= 10
        ).sum().item()


# --------------------------------------------------
# 6. Calculate digit accuracy
# --------------------------------------------------

accuracy = 100.0 * correct_digits / total_digits

print("\n" + "=" * 55)
print("MNIST DIGIT EVALUATION")
print("=" * 55)

print(f"Correct digit predictions: {correct_digits}/{total_digits}")
print(f"Overall digit accuracy: {accuracy:.2f}%")

print(
    "Digit images predicted as operators:",
    predicted_as_operator
)


# --------------------------------------------------
# 7. Digit-only report
# --------------------------------------------------

# Keep only images predicted as one of the digit classes
# for this digit-to-digit classification report.
digit_true = []
digit_pred = []

for true_idx, pred_idx in zip(true_labels, predicted_labels):
    if pred_idx < 10:
        digit_true.append(str(true_idx))
        digit_pred.append(CLASSES[pred_idx])

print("\nDigit classification report:")
print(
    classification_report(
        [str(x) for x in true_labels],
        [
            CLASSES[p] if p < len(CLASSES) else "?"
            for p in predicted_labels
        ],
        labels=DIGITS,
        zero_division=0,
        digits=4
    )
)


# --------------------------------------------------
# 8. Confusion matrix for all 14 classes
# --------------------------------------------------

print("\nConfusion matrix (true class rows, predicted columns):")
print(
    confusion_matrix(
        [CLASSES[t] for t in true_labels],
        [CLASSES[p] for p in predicted_labels],
        labels=CLASSES
    )
)

print("\nClass order:")
print(CLASSES)

