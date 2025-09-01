# 📸 Photo Processor

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)]()

A local, AI-powered tool to **automatically sort your vacation photos** into:  
- ✅ `best/` (sharp, aesthetic photos)  
- ❌ `bad-focus/` (blurry or out-of-focus)  
- 🤔 `undecided/` (if the model isn’t confident)  

Supports **JPG, PNG, and Nikon `.NEF` RAW files**. Runs fully **offline** to save costs and protect privacy.  

---

## ⚙️ Setup

### 1. Create a conda environment and activate it
```bash
conda create --name photo-processor python=3.10
conda activate photo-processor
```

### 2. Install Dependencies
# PyTorch core (CPU version by default)
```bash
pip install torch torchvision torchaudio
```

# Hugging Face Transformers (CLIP model + processor)
```bash
pip install transformers
```

# Extra dependency for some vision models
```bash
pip install timm
```

# Image handling (JPG/PNG)
```bash
pip install pillow
```

# Blur detection
```bash
pip install opencv-python
```

# RAW photo support (e.g., Nikon .NEF)
```bash
pip install rawpy
```

# Progress bar
```bash
pip install tqdm
```

### 3. Edit process_photos.py and set your photo directory:
input_dir = "photos" # replace this with your directory name

### 4. Run the processor:
```bash
python process_photos.py
```
Photos will be automatically moved into subfolders inside your photo directory:
best/
bad-focus/
undecided/

📂 Example Folder Structure

Before running:

photos/
├── IMG_001.NEF
├── IMG_002.JPG
├── IMG_003.NEF
├── IMG_004.PNG
└── IMG_005.JPG


After running:

photos/
├── best/
│   ├── IMG_002.JPG
│   └── IMG_005.JPG
├── bad-focus/
│   ├── IMG_001.NEF
│   └── IMG_003.NEF
├── undecided/
│   └── IMG_004.PNG

📝 Notes

Blur detection: Uses variance of Laplacian to catch out-of-focus photos.

Aesthetic scoring: Uses a CLIP-based regressor to rank images by visual quality.

Multi-threaded: Processes photos faster using multiple threads.

RAW support: Handles Nikon .NEF files and converts them to RGB for scoring and blur detection.

💡 Tips for Best Results

Works best on batches of recent vacation photos you want to quickly clean up and organize.

Adjust SHARPNESS_THRESHOLD in process_photos.py if your photos are unusually sharp or soft.

You can modify the aesthetic score thresholds (0.6 for best, 0.3 for undecided) to suit your taste.