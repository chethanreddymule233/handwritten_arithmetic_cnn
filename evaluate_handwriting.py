
from pathlib import Path
import csv

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import ArithmeticCNN, CLASSES


# --------------------------------------------------
# 1. Paths and class labels
# --------------------------------------------------

ROOT = Path("handwritten_test")
CHECKPOINT = Path("artifacts/arithmetic_cnn.pt")
OUTPUT = Path("artifacts")

OPERATOR_CLASSES = ["+", "-", "x", "/"]

FOLDER_NAMES = {
    "+": "plus",
    "-": "minus",
    "x": "x",
    "/": "slash",
}


# --------------------------------------------------
# 2. Check files and select device
# --------------------------------------------------

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}. Train the model first."
    )

if not ROOT.exists():
    raise FileNotFoundError(
        f"Test dataset folder not found: {ROOT}"
    )

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

model = ArithmeticCNN(num_classes=len(CLASSES))

if "classes" in checkpoint and checkpoint["classes"] != CLASSES:
    raise ValueError(
        "Class order mismatch!\n"
        f"Checkpoint: {checkpoint['classes']}\n"
        f"Current model: {CLASSES}"
    )

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Loaded checkpoint:", CHECKPOINT)
print("Classes:", CLASSES)


# --------------------------------------------------
# 4. Preprocess images
# --------------------------------------------------

# Match the real-operator dataset pipeline used in training:
# grayscale -> invert -> ToTensor.
transform = transforms.ToTensor()


def preprocess(path):
    image = Image.open(path).convert("L")
    image = ImageOps.invert(image)

    tensor = transform(image).unsqueeze(0)
    return tensor


# --------------------------------------------------
# 5. Evaluate the held-out operator images
# --------------------------------------------------

true_labels = []
predicted_labels = []
rows = []

for label in OPERATOR_CLASSES:
    folder = ROOT / FOLDER_NAMES[label]

    if not folder.exists():
        print(f"Missing folder: {folder}")
        continue

    files = sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp"
        }
    )

    print(f"{label}: {len(files)} images")

    for path in files:
        try:
            tensor = preprocess(path).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)

                predicted_index = probabilities.argmax(dim=1).item()
                predicted_class = CLASSES[predicted_index]
                confidence = probabilities[0, predicted_index].item()

            true_labels.append(label)
            predicted_labels.append(predicted_class)

            rows.append({
                "filename": str(path),
                "true_label": label,
                "predicted_label": predicted_class,
                "confidence": f"{confidence:.4f}",
                "correct": predicted_class == label
            })

        except Exception as exc:
            print(f"Could not process {path}: {exc}")


# --------------------------------------------------
# 6. Report results
# --------------------------------------------------

if not rows:
    raise RuntimeError(
        "No images were evaluated. Check the folders "
        "handwritten_test/plus, minus, x, and slash."
    )

correct = sum(row["correct"] for row in rows)
total = len(rows)
accuracy = 100.0 * correct / total

print("\n" + "=" * 55)
print("HANDWRITTEN OPERATOR EVALUATION")
print("=" * 55)

print(f"\nCorrect predictions: {correct}/{total}")
print(f"Overall accuracy: {accuracy:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=OPERATOR_CLASSES,
        zero_division=0,
        digits=4
    )
)

print("Confusion matrix (rows=true, columns=predicted):")
print(
    confusion_matrix(
        true_labels,
        predicted_labels,
        labels=OPERATOR_CLASSES
    )
)


# --------------------------------------------------
# 7. Save detailed predictions
# --------------------------------------------------

OUTPUT.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT / "handwritten_predictions.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "true_label",
            "predicted_label",
            "confidence",
            "correct"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDetailed predictions saved to: {csv_path}")

