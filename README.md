# Books Recognition System

## Project Overview

This project implements a computer vision system designed to recognize and localize books on a shelf using reference images (templates). Developed as part of the **Image Processing and Computer Vision** course, the system utilizes both traditional and modern deep learning-based techniques to detect multiple instances of books, providing their bounding boxes, dimensions, and positions.

## Authors

- **Pietro Tellarini**
- **Giorgia Bravi**

## Key Features

- **SOTA Pipeline**: Modern deep learning-based feature extraction (SuperPoint + LightGlue) with classical geometric verification
- **Traditional CV Pipeline**: SIFT-based feature extraction with geometric validation as fallback
- **OCR Verification**: Optional text-based verification of detections
- **Iterative Recognition (Spatial Peeling)**: Robust multi-object detection handling overlapping instances
- **Modular Architecture**: Clean class-based design with configurable components

---

## SOTA Pipeline Architecture

```
[Input RAW]
     |
     v
[IMAGE PROCESSING PIPELINE]
(CLAHE -> Denoise -> Sharpen)
     |
     v
[COMPUTER VISION PIPELINE]
(SuperPoint -> LightGlue -> DEGENSAC -> OCR Verification)
     |
     v
[Final Result]
(Book Found: YES/NO + Position)
```

### Quick Start

```python
from pipeline import BookRecognitionPipeline

# Initialize pipeline
pipeline = BookRecognitionPipeline()

# Load book models
pipeline.load_library("dataset/models")

# Process a scene
results = pipeline.process("dataset/scenes/scene_01.jpg")

# Get results
for detection in results.detections:
    print(f"Found {detection.book_id} at {detection.bbox.center}")
```

### Command Line Usage

```bash
# Process all scenes with default settings
python run_pipeline.py

# Process specific scene
python run_pipeline.py --scene scene_01.jpg

# Use fast preset (speed over accuracy)
python run_pipeline.py --fast

# Use fallback methods (SIFT + FLANN, no deep learning)
python run_pipeline.py --fallback

# Save visualizations
python run_pipeline.py --visualize
```

### Pipeline Components

| Component | SOTA Method | Fallback |
|-----------|-------------|----------|
| Feature Extraction | SuperPoint | SIFT + RootSIFT |
| Feature Matching | LightGlue | FLANN + Lowe's Ratio |
| Geometric Verification | DEGENSAC | OpenCV RANSAC |
| OCR Verification | EasyOCR | Tesseract |

---

## Technical Details

### 1. Image Processing Pipeline

The preprocessing pipeline enhances images for optimal feature extraction:

- **CLAHE**: Contrast Limited Adaptive Histogram Equalization on LAB L-channel
- **Denoising**: Non-local means denoising for noise reduction
- **Sharpening**: Unsharp mask for edge enhancement

### 2. Feature Extraction (SuperPoint)

- **SuperPoint**: Deep learning-based keypoint detection with learned descriptors
- **Parameters**:
  - `max_keypoints`: 2048 (configurable)
  - `nms_radius`: 4 pixels
  - `detection_threshold`: 0.005
- **Fallback**: SIFT with RootSIFT normalization

### 3. Feature Matching (LightGlue)

- **LightGlue**: Attention-based feature matching optimized for SuperPoint
- **Benefits**:
  - Robust to viewpoint changes
  - Handles wide baselines
  - No ratio test needed
- **Fallback**: FLANN with Lowe's ratio test

### 4. Geometric Verification (DEGENSAC)

- **DEGENSAC**: Handles degenerate (planar) configurations common in book matching
- **Validation**:
  - Homography determinant check
  - Condition number validation
  - Bounding box convexity and area checks
  - Aspect ratio verification

### 5. OCR Verification (Optional)

- Extracts and rectifies detected book regions
- Compares OCR text with expected book titles
- Uses similarity metrics (sequence matching + word overlap)

---

## Legacy Pipeline

The original SIFT-based approach is still available:

## Structure

```
BooksRecognition/
├── pipeline/                    # SOTA Pipeline Module
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration classes
│   ├── image_processor.py      # CLAHE + Denoise + Sharpen
│   ├── feature_extractor.py    # SuperPoint / SIFT
│   ├── feature_matcher.py      # LightGlue / FLANN
│   ├── geometric_verifier.py   # DEGENSAC homography
│   ├── ocr_verifier.py         # EasyOCR / Tesseract
│   ├── book_recognition.py     # Main pipeline orchestrator
│   └── result.py               # Result data classes
├── run_pipeline.py             # CLI demo script
├── alternative_2.py            # Legacy hybrid pipeline
├── assignement.py              # Legacy DBSCAN + peeling
├── assignment_module_one.ipynb # Development notebook
├── dataset/
│   ├── models/                 # Book template images
│   └── scenes/                 # Scene images to process
├── requirements.txt            # Dependencies
└── README.md
```

## Requirements

### SOTA Pipeline (Recommended)
```bash
pip install torch torchvision  # PyTorch for GPU support
pip install kornia              # SuperPoint + LightGlue
pip install pydegensac          # DEGENSAC
pip install easyocr             # OCR verification
pip install opencv-python numpy matplotlib scikit-learn
```

### Minimal (Fallback Mode)
```bash
pip install opencv-python numpy matplotlib
```

See [requirements.txt](requirements.txt) for full dependencies.
