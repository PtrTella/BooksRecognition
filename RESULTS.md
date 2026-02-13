# Book Spine Recognition — Experimental Results

## Dataset

- **22 model images** (book covers, `model_0.png` to `model_21.png`)
- **29 scene images** (bookshelves, `scene_0.jpg` to `scene_28.jpg`)
- **57 total ground truth book instances** across all scenes
- Challenges: multiple identical copies, thin spines, varying lighting, occlusion

---

## Results Summary

### Best Configuration: SIFT + DBSCAN + NCC Verification (alternative.py)

| Metric | Value |
|--------|-------|
| **F1 Score** | **0.926** |
| Precision | 0.980 |
| Recall | 0.877 |
| TP / FP / FN | 50 / 1 / 7 |

### Full Comparison Table

| Pipeline | Precision | Recall | F1 | TP | FP | FN | Notes |
|----------|-----------|--------|------|----|----|-----|-------|
| **SIFT + DBSCAN + NCC** | **0.980** | **0.877** | **0.926** | 50 | 1 | 7 | Best overall |
| SIFT + DBSCAN (no NCC) | 0.708 | 0.895 | 0.791 | 51 | 21 | 6 | Most TPs but too many FPs |
| SIFT + Hierarchical | 0.825 | 0.825 | 0.825 | 47 | 10 | 10 | Previous best (debug) |
| SIFT + LightGlue | 0.815 | 0.772 | 0.793 | 44 | 10 | 13 | Learned matcher, no improvement |
| DISK + LightGlue | 0.853 | 0.509 | 0.637 | 29 | 5 | 28 | Good precision, low recall |
| KeyNet + HardNet | 1.000 | 0.456 | 0.627 | 26 | 0 | 31 | Perfect precision, poor recall |
| LoFTR | 0.536 | 0.649 | 0.587 | 37 | 32 | 20 | Dense matcher, many FPs |
| DeDoDe | 0.000 | 0.000 | 0.000 | 0 | 0 | 57 | Too few matches |

---

## Pipeline Architecture (Best Config)

```
Input Image
    │
    ▼
Preprocessing (CLAHE + Unsharp Mask on LAB L-channel)
    │
    ▼
SIFT/RootSIFT Feature Extraction (contrast=0.01, edge=15, octaves=4)
    │
    ▼
FLANN Matching + Lowe Ratio Test (ratio=0.78) + MAD Scale Filter (σ=3.5)
    │
    ▼
Orientation-Aware Star-Model Voting (GHT centre projection)
    │
    ▼
DBSCAN Clustering (eps=20, min_pts=3)
    │
    ▼
Geometric Verification — Affine Peeling (min_inliers=3, RANSAC=4.0px)
    │
    ▼
NMS (iou_same=0.30, iou_cross=0.35)
    │
    ▼
NCC Warp-and-Compare Verification (threshold=0.20)     ← KEY INNOVATION
    │
    ▼
Final Detections
```

---

## Key Findings

### 1. Classical SIFT dominates deep learning features for this task

SIFT keypoints provide per-keypoint **orientation** and **scale**, which are essential
for the star-model (GHT) centre-vote projection. Without these properties, deep
features cannot properly separate multiple instances of identical books.

| Property | SIFT | DISK | KeyNet | LoFTR | DeDoDe |
|----------|------|------|--------|-------|--------|
| Orientation | ✓ | ✗ | ✓ (AffNet) | ✗ | ✗ |
| Scale | ✓ | ✗ | ✓ (AffNet) | ✗ | ✗ |
| Multi-instance | ✓✓ | ✗ | ✓ | ✗ | ✗ |
| Texture discrimination | ✓✓ | ✓ | ✓ | ✗ | ✗ |

### 2. NCC Verification is the breakthrough technique

The NCC (Normalised Cross Correlation) warp-and-compare step provides
near-perfect separation between true and false detections:

| Metric | True Positives | False Positives |
|--------|----------------|-----------------|
| NCC median | 0.829 | 0.073 |
| NCC mean | 0.711 | 0.134 |
| NCC range | [0.015, 0.956] | [-0.068, 0.585] |

At NCC threshold = 0.20:

- Removes **20 out of 21 FP** (95.2% FP rejection)  
- Loses only **1 out of 51 TP** (1.96% TP loss)
- Net F1 improvement: **0.791 → 0.926** (+17.1%)

### 3. DBSCAN outperforms hierarchical clustering

DBSCAN with eps=20, min_pts=3 finds more true instances (51 TP) than
hierarchical clustering (47 TP), especially on scenes with multiple
identical books (scene_17: 5/5 vs 3/5 detections).

### 4. LightGlue is NOT better than FLANN for this task

Replacing FLANN ratio-test matching with LightGlue learned matching
actually **decreased** performance (F1: 0.825 → 0.793). LightGlue
produces fewer matches overall and doesn't benefit from SIFT's
orientation/scale information as effectively as FLANN + ratio test + MAD filter.

---

## NCC Threshold Ablation

| NCC Threshold | TP | FP | FN | P | R | F1 |
|---------------|----|----|-----|-------|-------|-------|
| None | 51 | 21 | 6 | 0.708 | 0.895 | 0.791 |
| 0.10 | 51 | 9 | 6 | 0.850 | 0.895 | 0.872 |
| 0.15 | 51 | 6 | 6 | 0.895 | 0.895 | 0.895 |
| 0.18 | 50 | 3 | 7 | 0.943 | 0.877 | 0.909 |
| **0.20** | **50** | **1** | **7** | **0.980** | **0.877** | **0.926** |
| 0.25 | 50 | 1 | 7 | 0.980 | 0.877 | 0.926 |
| 0.30 | 46 | 1 | 11 | 0.979 | 0.807 | 0.885 |

---

## Remaining Errors (Best Config)

### False Positive (1)

- **scene_18**: Extra model_9 detected (NCC=0.585, above threshold)

### False Negatives (7)

- **scene_10**: 2 of 4 model_19 copies missed (multi-instance limitation)
- **scene_18**: 1 model missing (complex scene, 6 books total)
- **scene_27**: 1 model_3 missed
- **scene_9**: 3 of 4 model_19 copies missed (challenging viewpoint)

---

## Optimal Parameters (alternative.py)

```python
Config(
    # Preprocessing
    clahe_clip=3.0, sharpen_amount=1.0,
    
    # SIFT
    contrast_threshold=0.01, edge_threshold=15, n_octave_layers=4,
    root_sift=True,
    
    # Matching
    lowe_ratio=0.78, scale_sigma=3.5,
    
    # Clustering
    dbscan_eps=20.0, dbscan_min_pts=3,
    
    # Verification  
    ransac_thresh=4.0, min_inliers=3, min_area=450,
    min_det=0.025, max_det=40.0, ar_tolerance=3.0,
    
    # NMS
    iou_same=0.30, iou_cross=0.35,
    
    # Post-verification (KEY)
    ncc_verify=True, ncc_threshold=0.20,
    edge_verify=True, edge_tolerance=2.6,
)
```

## Files

| File | Purpose | Best F1 |
|------|---------|---------|
| `alternative.py` | Production pipeline with DBSCAN + NCC verify | **0.926** |
| `assignement_debug.py` | Debug pipeline with hierarchical clustering | 0.825 |
| `sota.py` | Deep learning backends (DISK, KeyNet, LoFTR, DeDoDe) | 0.637 |
| `hybrid.py` | SIFT+LightGlue and ALIKED experiments | 0.793 |
| `ncc_analysis.py` | NCC distribution analysis script | — |
