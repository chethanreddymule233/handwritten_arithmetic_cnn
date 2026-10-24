import ast
import operator
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from model import ArithmeticCNN, CLASSES

MODEL_PATH = Path("artifacts/arithmetic_cnn_handwritten_x.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

st.set_page_config(page_title="Custom CNN Arithmetic Reader", page_icon="🧮", layout="wide")
st.title("Handwritten Arithmetic Recognition")
st.write("A custom CNN trained from scratch recognizes digits and +, -, x, / symbols; a separate parser evaluates the expression.")
st.warning("Prototype limitations: one horizontal expression per image, symbols should be separated, and predictions can be wrong. Verify the recognized expression before trusting the answer.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = ArithmeticCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def prepare_symbol(binary_crop):
    """Input is a crop with white foreground strokes on black background."""
    ys, xs = np.where(binary_crop > 0)
    if len(xs) == 0:
        return None
    crop = binary_crop[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - nw) // 2, (28 - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor




def segment_expression(pil_image):
    rgb = np.array(ImageOps.exif_transpose(pil_image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 11
    )
    H, W = binary.shape

    # Remove tiny components, while preserving thin horizontal strokes.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    min_area = max(5, int(H * W * 0.00003))

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        is_horizontal_stroke = (
            w >= max(5, int(W * 0.01))
            and w >= 2 * max(h, 1)
        )
        if area >= min_area or is_horizontal_stroke:
            cleaned[labels == i] = 255

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )


    min_h = max(2, int(H * 0.015))
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Preserve thin horizontal symbols such as minus signs.
        is_horizontal_stroke = (
            w >= max(5, int(W * 0.01))
            and w >= 2 * max(h, 1)
        )

        if (h >= min_h and w >= 2) or is_horizontal_stroke:
            boxes.append((x, y, w, h))

    # Group detected components into rows by vertical center.
    boxes.sort(key=lambda b: (b[1] + b[3] / 2, b[0]))
    rows = []

    for box in boxes:
        x, y, w, h = box
        center_y = y + h / 2
        best_row = None
        best_distance = float("inf")

        for row in rows:
            row_center = np.median([b[1] + b[3] / 2 for b in row])
            row_height = np.median([b[3] for b in row])
            distance = abs(center_y - row_center)

            if distance <= max(10, 0.6 * max(h, row_height)):
                if distance < best_distance:
                    best_row = row
                    best_distance = distance

        if best_row is None:
            rows.append([box])
        else:
            best_row.append(box)

    rows.sort(key=lambda row: np.median([b[1] + b[3] / 2 for b in row]))
    symbols = []

    for row_index, row in enumerate(rows):
        row.sort(key=lambda b: b[0])
        merged = []

        for box in row:
            x, y, w, h = box

            if merged:
                px, py, pw, ph = merged[-1]
                gap = x - (px + pw)
                overlap = min(y + h, py + ph) - max(y, py)

                if (
                    gap <= max(3, int(0.025 * W))
                    and overlap > -0.25 * min(h, ph)
                ):
                    x0, y0 = min(x, px), min(y, py)
                    x1, y1 = max(x + w, px + pw), max(y + h, py + ph)
                    merged[-1] = (x0, y0, x1 - x0, y1 - y0)
                    continue

            merged.append(box)

        for x, y, w, h in merged:
            pad = max(3, int(0.12 * max(w, h)))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = cleaned[y0:y1, x0:x1]
            symbols.append((crop, (x, y, w, h), row_index))

    return rgb, cleaned, symbols

def predict_symbol(model, crop):
    tensor = prepare_symbol(crop)
    if tensor is None:
        return "?", 0.0, None
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(probs.argmax().item())
    return CLASSES[idx], float(probs[idx].item()), probs.cpu().numpy()

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
def safe_eval(expression):
    expression = expression.replace("×", "*").replace("x", "*").replace("X", "*")
    tree = ast.parse(expression, mode="eval")
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("Division by zero is undefined.")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = visit(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        raise ValueError("Only digits and +, -, x/×, / are supported in this prototype.")
    result = visit(tree)
    return int(result) if isinstance(result, float) and result.is_integer() else result

model = load_model()
if model is None:
    st.error("Model file not found. Train the model first: `python train.py --epochs 8`")
    st.stop()

uploaded = st.file_uploader("Upload one expression image", type=["png", "jpg", "jpeg", "bmp"])
st.caption("Tip: use dark handwriting on a light background. Keep symbols on one line and leave a little space between them.")
if uploaded:
    pil = Image.open(uploaded)
    rgb, binary, symbols = segment_expression(pil)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Input and preprocessing")
        st.image(rgb, caption="Uploaded image", use_container_width=True)
        st.image(binary, caption="Thresholded image (white foreground)", clamp=True, use_container_width=True)

    if not symbols:
        st.error("No symbols detected. Try a clearer image with stronger contrast.")
        st.stop()

    predictions = []
    cols = st.columns(min(len(symbols), 7))

    # Predict every symbol and retain its row index.
    for i, (crop, box, row_index) in enumerate(symbols):
        label, confidence, probs = predict_symbol(model, crop)
        predictions.append((label, confidence, crop, box, row_index))

        with cols[i % len(cols)]:
            st.image(
                crop,
                caption=f"Symbol {i+1} — segmented crop",
                clamp=True,
                use_container_width=True
            )
            st.write(f"**Prediction: {label}**")
            st.caption(f"Confidence: {confidence:.1%}")

            # Show the exact 28x28 input used by the CNN.
            processed = prepare_symbol(crop)
            if processed is not None:
                processed_img = (
                    processed.squeeze(0).squeeze(0).cpu().numpy() * 255
                ).round().astype("uint8")

                st.image(
                    processed_img,
                    caption=f"Symbol {i+1} — CNN input (28x28)",
                    clamp=True,
                    width=112
                )

            # Show the three most likely classes.
            if probs is not None:
                probabilities = np.asarray(probs).flatten()
                top_indices = probabilities.argsort()[-3:][::-1]

                st.caption("Top predictions:")
                for idx in top_indices:
                    st.write(f"{CLASSES[idx]}: {probabilities[idx]:.1%}")

    threshold = st.slider(
        "Minimum confidence to accept a symbol",
        0.30, 0.99, 0.70, 0.01
    )

    st.subheader("Recognition results")

    # Group predictions by their detected row.
    row_indices = sorted(set(p[4] for p in predictions))

    for row_number, row_index in enumerate(row_indices, start=1):
        row_predictions = [p for p in predictions if p[4] == row_index]
        row_predictions.sort(key=lambda p: p[3][0])

        recognized = "".join(p[0] for p in row_predictions)

        st.markdown(f"**Expression {row_number}**")
        st.code(recognized if recognized else "(nothing recognized)")

        low = [
            i + 1
            for i, p in enumerate(row_predictions)
            if p[1] < threshold
        ]

        if low:
            st.warning(
                f"Expression {row_number}: low-confidence symbol(s) "
                f"{low}. Check these predictions before trusting the answer."
            )

        try:
            answer = safe_eval(recognized)
            st.metric(f"Answer — Expression {row_number}", str(answer))
        except ZeroDivisionError as e:
            st.error(f"Expression {row_number}: {e}")
        except Exception as e:
            st.error(
                f"Could not evaluate expression {row_number}: {e}"
            )
            st.info(
                "Supported operators are +, -, x or × for multiplication, "
                "and / for division. Use / rather than ÷."
            )

    st.caption(
        "Verify each recognized expression before trusting its answer. "
        "Incorrect symbol predictions can produce incorrect calculations."
    )

