# Product Recognition of Books

## Overview
This repository contains a computer vision system developed as an assignment for the **Image Processing and Computer Vision** course. The goal is to identify books on a shelf given reference images (models) of the books.

The system uses **traditional computer vision techniques** to detect multiple instances of book spines or covers in scene images, reporting their location, area, and bounding boxes.

## Pipeline
The recognition pipeline consists of the following stages:
1. **Preprocessing:** CLAHE on L-channel of CIE-LAB, USM sharpening, and DoG band-pass filter.
2. **Feature Extraction:** Root-SIFT descriptors.
3. **Feature Matching:** FLANN-based matching with Lowe's ratio test.
4. **StarVoter:** Generalized Hough Transform for instance hypothesis.
5. **Clustering:** DBSCAN to group matches from the same instance.
6. **Geometric Verification:** RANSAC for homography estimation and inlier counting.
7. **NMS (Non-Maximum Suppression):** To handle overlapping detections.
8. **NCC Verifier:** Normalized Cross-Correlation for final validation.

## Authors
- **Pietro Tellarini** - [pietro.tellarini2@studio.unibo.it](mailto:pietro.tellarini2@studio.unibo.it)
- **Giorgia Bravi** - [giorgia.bravi2@studio.unibo.it](mailto:giorgia.bravi2@studio.unibo.it)

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
The main implementation and experimental results are located in:
- `assignment_module_one.ipynb`
