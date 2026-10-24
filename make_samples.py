"""Create font-rendered sample expressions for testing image upload and segmentation.
These are synthetic pipeline tests, not handwritten validation data."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

OUT = Path("sample_images")
OUT.mkdir(exist_ok=True)
font_candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
font_path = next((p for p in font_candidates if Path(p).exists()), None)
examples = ["7 + 2", "8 x 3", "9 - 4", "6 + 5 x 2"]
for i, text in enumerate(examples, 1):
    img = Image.new("RGB", (520, 150), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 68) if font_path else ImageFont.load_default()
    draw.text((30, 28), text, fill="black", font=font)
    img.save(OUT / f"expression_{i}.png")
print(f"Saved synthetic examples to {OUT.resolve()}")
print("These verify the upload pipeline only; use real handwriting for meaningful evaluation.")
