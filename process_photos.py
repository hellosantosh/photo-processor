import os
import csv
import shutil
import torch
import torch.nn as nn
import cv2
import numpy as np
import rawpy
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPProcessor, CLIPModel


# ------------------------
# Aesthetic Regressor
# ------------------------
class AestheticRegressor(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.regressor = nn.Linear(self.clip.vision_model.config.hidden_size, 1)

    def forward(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output
        score = self.regressor(pooled_output)
        return score


# ------------------------
# Setup
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = AestheticRegressor().to(device).eval()

input_dir = "photos"
output_dirs = {
    "best": os.path.join(input_dir, "best"),
    "bad-focus": os.path.join(input_dir, "bad-focus"),
    "undecided": os.path.join(input_dir, "undecided"),
}
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

csv_file = os.path.join(input_dir, "photo_scores.csv")


# ------------------------
# Blur detection helper
# ------------------------
def detect_blur(img_array, threshold=100.0):
    """Return True if image is blurry based on variance of Laplacian."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm


# ------------------------
# Image loader (handles NEF/RAW and JPEG/PNG)
# ------------------------
def load_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".nef", ".raw"]:  # Nikon RAW
        with rawpy.imread(image_path) as raw:
            rgb = raw.postprocess()
        img_array = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(rgb)
        return pil_image, rgb
    else:
        pil_image = Image.open(image_path).convert("RGB")
        img_array = np.array(pil_image)
        return pil_image, img_array


# ------------------------
# Scoring function
# ------------------------
def process_image(image_path):
    try:
        pil_image, img_array = load_image(image_path)

        # Blur detection
        is_blurry, blur_value = detect_blur(img_array)

        if is_blurry:
            category = "bad-focus"
            score = None
        else:
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                score = model(inputs["pixel_values"])
                score = float(score.item())

            if score >= 0.6:
                category = "best"
            else:
                category = "undecided"

        # Move file instead of copy
        dest_path = os.path.join(output_dirs[category], os.path.basename(image_path))
        shutil.move(image_path, dest_path)

        return (os.path.basename(image_path), score, blur_value, category)

    except Exception as e:
        return (os.path.basename(image_path), None, None, f"error: {e}")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    image_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".nef", ".raw"))
    ]

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_image, img): img for img in image_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing photos"):
            results.append(future.result())

    # Write CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "aesthetic_score", "blur_value", "category"])
        writer.writerows(results)

    print(f"\n✅ Done! Sorted photos into {list(output_dirs.keys())}")
    print(f"📊 Scores + blur metrics saved to {csv_file}")
