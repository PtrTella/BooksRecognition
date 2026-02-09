# Books Recognition System

## Project Overview

This project implements a computer vision system designed to recognize and localize books on a shelf using reference images (templates). Developed as part of the **Image Processing and Computer Vision** course, the system utilizes traditional computer vision techniques to detect multiple instances of books, providing their bounding boxes, dimensions, and positions.

## Authors

- **Pietro Tellarini**
- **Giorgia Bravi**

## Key Features

- **Traditional CV Pipeline**: No deep learning; relies on feature extraction and geometric validation.
- **Iterative Recognition (Spatial Peeling)**: A robust multi-object detection strategy that handles overlapping and multiple instances by iteratively "peeling" away identified keypoints.
- **Optimized Pre-processing**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and Unsharp Masking to enhance feature detection in various lighting conditions.
- **Geometric Validation**: Filters out false positives using area constraints, rectangularity checks, and homography validation.

## Technical Details

### 1. Pre-processing

The system utilizes a `Grayscale (CLAHE) + Unsharp Masking` pipeline.

- **CLAHE**: Increases keypoint detection by ~34% and improves stability across lighting variations.
- **Unsharp Masking**: Enhances typographic edges, allowing more precise SIFT gradient orientations.

### 2. Feature Extraction & Matching

- **Algorithm**: SIFT (Scale-Invariant Feature Transform) with RootSIFT normalization.
- **Parameters**:
  - `nfeatures`: 10,000 (to capture detail on thin spines).
  - `lowe_ratio`: 0.90 (optimized for high recall and handling similar-looking books).
  - `ransac_thresh`: 8.0 pixel tolerance for homography estimation.

### 3. Iterative Recognizer Architecture

The `IterativeBookRecognizer` class handles the complex task of multi-object detection:

1. **Competitive Matching**: Matches all templates against remaining scene keypoints.
2. **Best Candidate Selection**: Selects the detection with the highest inlier count.
3. **Spatial Peeling**: Removes keypoints falling within the detected bounding box from the pool for the next iteration.

### 4. Discarded Approaches

The project also explored a `LibrarySegmenter` (Sobel gradients + column profiling), which was ultimately discarded due to excessive computational overhead and lower flexibility compared to the iterative keypoint-based approach.

## Structure

- [assignment_module_one.ipynb](assignment_module_one.ipynb): Main development and documentation notebook.
- `dataset/`: Contains reference `models` and shelf `scenes`.
- `sketch/`: Experimental scripts and discarded modules (e.g., [Segmenter.py](sketch/Segmenter.py)).

## Requirements

See [requirements.txt](requirements.txt) for dependencies (OpenCV, NumPy, Matplotlib, etc.).
