
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import ArithmeticCNN, CLASSES
from app import load_model, prepare_symbol, segment_expression

# Use the same model and preprocessing as the app.
model = load_model()
if model is None:
    raise RuntimeError("Could not load the model.")

image_path = Path("handwritten_test_expression.png")

if not image_path.exists():
    raise FileNotFoundError(
        "Save your 9 x 2 test photo as handwritten_test_expression.png "
        "in the project folder first."
    )

image = Image.open(image_path)
_, _, symbols = segment_expression(image)

for i, (crop, box) in enumerate(symbols):
    tensor = prepare_symbol(crop)
    if tensor is None:
        continue

    device = next(model.parameters()).device
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor.to(device)), dim=1)[0]
        top = torch.topk(probabilities, k=5)

    print(f"\nSymbol {i + 1}, bounding box={box}")
    for idx, probability in zip(top.indices.cpu().tolist(),
                                top.values.cpu().tolist()):
        print(f"  {CLASSES[idx]}: {probability:.2%}")

