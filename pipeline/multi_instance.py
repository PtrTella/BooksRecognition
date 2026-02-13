"""
Book Spine Recognition — Best Pipeline (F1 = 0.926)
=====================================================

SIFT/RootSIFT  →  FLANN + Lowe ratio + MAD scale filter
              →  Star-Model GHT centre voting
              →  DBSCAN spatial clustering
              →  Affine peeling per cluster
              →  NMS (two-tier IoU)
              →  NCC warp-and-compare verification

Run:
    python pipeline/multi_instance.py

Output images are saved to  output_multiinstance/
A summary figure with all scenes is displayed at the end.

Architecture
------------
┌─────────────┐    ┌───────────────┐    ┌──────────────┐
│ Preprocessor │───▶│FeatureExtract.│───▶│FeatureMatcher│
│  CLAHE+USM   │    │  RootSIFT     │    │ FLANN+Lowe   │
└─────────────┘    └───────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼───────┐
                    │  Clusterer   │◀───│  StarVoter    │
                    │   DBSCAN     │    │ GHT centre    │
                    └──────┬───────┘    └───────────────┘
                           │
                    ┌──────▼───────┐    ┌───────────────┐
                    │  Geometric   │───▶│     NMS       │
                    │  Verifier    │    │  IoU 2-tier   │
                    │  (peeling)   │    └──────┬────────┘
                    └──────────────┘           │
                                        ┌─────▼────────┐
                                        │ NCC Verifier  │
                                        │ warp+compare  │
                                        └──────┬───────┘
                                               │
                                        ┌──────▼───────┐
                                        │  Visualizer   │
                                        │  draw + save  │
                                        └──────────────┘
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# ──────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "dataset" / "models"
SCENES_DIR = ROOT / "dataset" / "scenes"
OUTPUT_DIR = ROOT / "output_multiinstance"

# ──────────────────────────────────────────────────────────────────────────
#  Ground Truth
# ──────────────────────────────────────────────────────────────────────────
GROUND_TRUTH: Dict[str, List[str]] = {
    "scene_0": [],
    "scene_1": ["model_18", "model_18"],
    "scene_10": ["model_19", "model_19", "model_19", "model_19"],
    "scene_11": [],
    "scene_12": [],
    "scene_13": [],
    "scene_14": [],
    "scene_15": ["model_11", "model_11", "model_12", "model_12", "model_12"],
    "scene_16": ["model_11", "model_11", "model_12", "model_12", "model_12"],
    "scene_17": ["model_11", "model_11", "model_12", "model_12", "model_12"],
    "scene_18": ["model_10", "model_10", "model_10", "model_8", "model_8", "model_9"],
    "scene_19": ["model_6", "model_6", "model_6", "model_7", "model_7"],
    "scene_2": ["model_17"],
    "scene_20": [],
    "scene_21": [],
    "scene_22": [],
    "scene_23": ["model_5"],
    "scene_24": [],
    "scene_25": [],
    "scene_26": ["model_0", "model_0", "model_4"],
    "scene_27": ["model_2", "model_2", "model_3", "model_3"],
    "scene_28": ["model_1", "model_1"],
    "scene_3": ["model_16", "model_16"],
    "scene_4": ["model_14", "model_14", "model_15", "model_15"],
    "scene_5": ["model_13"],
    "scene_6": ["model_21"],
    "scene_7": ["model_20", "model_20"],
    "scene_8": [],
    "scene_9": ["model_19", "model_19", "model_19", "model_19"],
}


# ══════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """All tuneable hyper-parameters — values tuned for F1 = 0.926."""

    # ── Preprocessing (CLAHE + unsharp-mask on LAB L-channel) ──
    clahe_clip:         float = 3.0
    clahe_grid:         int   = 8
    sharpen_amount:     float = 1.0

    # ── SIFT ──
    nfeatures:          int   = 0
    contrast_threshold: float = 0.01
    edge_threshold:     float = 15
    n_octave_layers:    int   = 4
    root_sift:          bool  = True

    # ── FLANN matching ──
    lowe_ratio:    float = 0.78
    flann_trees:   int   = 5
    flann_checks:  int   = 100
    scale_sigma:   float = 3.5

    # ── DBSCAN clustering ──
    dbscan_eps:      float = 20.0
    dbscan_min_pts:  int   = 3

    # ── Geometric verification ──
    ransac_thresh:   float = 4.0
    min_inliers:     int   = 3
    min_det:         float = 0.025
    max_det:         float = 40.0
    min_area:        int   = 450
    max_area_frac:   float = 0.9
    ar_tolerance:    float = 3.0

    # ── NMS ──
    iou_same:  float = 0.30
    iou_cross: float = 0.35

    # ── Edge-density post-filter ──
    edge_verify:    bool  = True
    edge_tolerance: float = 2.6

    # ── NCC warp-and-compare verification ──
    ncc_verify:     bool  = True
    ncc_threshold:  float = 0.20


# ══════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ══════════════════════════════════════════════════════════════════════════
class Preprocessor:
    """CLAHE on L-channel of CIE-LAB + unsharp mask → grayscale."""

    def __init__(self, cfg: Config) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
        )
        self._alpha = cfg.sharpen_amount

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch = self._clahe.apply(lab[:, :, 0])
        blur = cv2.GaussianBlur(l_ch, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(l_ch, 1.0 + self._alpha, blur, -self._alpha, 0)


# ══════════════════════════════════════════════════════════════════════════
#  Feature Extraction
# ══════════════════════════════════════════════════════════════════════════
class FeatureExtractor:
    """SIFT / RootSIFT keypoint + descriptor computation."""

    def __init__(self, cfg: Config) -> None:
        self._root = cfg.root_sift
        self._sift = cv2.SIFT_create(
            nfeatures=cfg.nfeatures,
            contrastThreshold=cfg.contrast_threshold,
            edgeThreshold=cfg.edge_threshold,
            nOctaveLayers=cfg.n_octave_layers,
        )

    def __call__(self, gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        kp, des = self._sift.detectAndCompute(gray, None)
        if des is None:
            return [], None
        des = des.astype(np.float32)
        if self._root:
            des /= des.sum(axis=1, keepdims=True) + 1e-7
            np.sqrt(des, out=des)
        return kp, des


# ══════════════════════════════════════════════════════════════════════════
#  Feature Matching
# ══════════════════════════════════════════════════════════════════════════
class FeatureMatcher:
    """Global FLANN index across all models.

    Lowe ratio test  +  MAD-based scale-ratio consistency filter.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._ids: List[str] = []
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=cfg.flann_trees),
            dict(checks=cfg.flann_checks),
        )

    def add(self, bid: str, des: np.ndarray) -> None:
        self._matcher.add([des])
        self._ids.append(bid)

    def build(self) -> None:
        self._matcher.train()

    def match(
        self,
        des_scene: Optional[np.ndarray],
        kp_scene: List[cv2.KeyPoint],
        model_kps: Dict[str, List[cv2.KeyPoint]],
    ) -> Dict[str, list]:
        if des_scene is None or not self._ids:
            return {}

        raw = self._matcher.knnMatch(des_scene, k=2)

        groups: Dict[str, list] = defaultdict(list)
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self._cfg.lowe_ratio * n.distance:
                groups[self._ids[m.imgIdx]].append(m)

        sigma = self._cfg.scale_sigma
        filtered: Dict[str, list] = {}
        for bid, ms in groups.items():
            if len(ms) < 3:
                filtered[bid] = ms
                continue
            mkp = model_kps.get(bid)
            if mkp is None:
                filtered[bid] = ms
                continue

            valid_ms, ratios = [], []
            for m in ms:
                if m.trainIdx < len(mkp):
                    valid_ms.append(m)
                    ratios.append(kp_scene[m.queryIdx].size / mkp[m.trainIdx].size)

            if len(valid_ms) < 3:
                filtered[bid] = ms
                continue

            ratios = np.array(ratios)
            med = np.median(ratios)
            mad = np.median(np.abs(ratios - med)) + 1e-7
            keep = np.abs(ratios - med) <= sigma * mad
            filtered[bid] = [m for m, k in zip(valid_ms, keep) if k]

        return filtered


# ══════════════════════════════════════════════════════════════════════════
#  Star-Model Voter (GHT centre-vote projection)
# ══════════════════════════════════════════════════════════════════════════
class StarVoter:
    """Stores displacement vectors (keypoint → model centre) and projects
    matched scene keypoints to centre-vote locations."""

    def __init__(self) -> None:
        self._templates: Dict[str, dict] = {}

    @property
    def templates(self) -> Dict[str, dict]:
        return self._templates

    def register(self, bid: str, kp: List[cv2.KeyPoint], shape: Tuple[int, ...]) -> None:
        h, w = shape[:2]
        cx, cy = w / 2.0, h / 2.0

        pts = np.array([k.pt for k in kp], dtype=np.float64)
        angles = np.deg2rad([k.angle for k in kp])
        sizes = np.array([k.size for k in kp])

        d = np.array([cx, cy]) - pts
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        vecs = np.column_stack([
             cos_a * d[:, 0] + sin_a * d[:, 1],
            -sin_a * d[:, 0] + cos_a * d[:, 1],
        ]) / sizes[:, np.newaxis]

        self._templates[bid] = {"kp": kp, "vecs": vecs, "shape": shape}

    def project(self, matches: list, kp_scene: List[cv2.KeyPoint], tpl: dict) -> np.ndarray:
        vecs = tpl["vecs"]
        qidx = [m.queryIdx for m in matches]
        tidx = [m.trainIdx for m in matches]

        pts = np.array([kp_scene[i].pt for i in qidx])
        angles = np.deg2rad([kp_scene[i].angle for i in qidx])
        sizes = np.array([kp_scene[i].size for i in qidx])
        stored = vecs[tidx]

        cos_a, sin_a = np.cos(angles), np.sin(angles)
        return pts + np.column_stack([
            cos_a * stored[:, 0] - sin_a * stored[:, 1],
            sin_a * stored[:, 0] + cos_a * stored[:, 1],
        ]) * sizes[:, np.newaxis]


# ══════════════════════════════════════════════════════════════════════════
#  DBSCAN Clusterer
# ══════════════════════════════════════════════════════════════════════════
class Clusterer:
    """DBSCAN spatial grouping of 2-D centre votes.  Noise (label -1) discarded."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def __call__(self, centres: np.ndarray) -> Dict[int, List[int]]:
        if len(centres) < 2:
            return {0: list(range(len(centres)))} if len(centres) else {}
        db = DBSCAN(eps=self._cfg.dbscan_eps, min_samples=self._cfg.dbscan_min_pts)
        labels = db.fit_predict(centres)
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            if label >= 0:
                clusters[label].append(i)
        return clusters


# ══════════════════════════════════════════════════════════════════════════
#  Geometric Verifier (affine peeling)
# ══════════════════════════════════════════════════════════════════════════
class GeometricVerifier:
    """Partial-affine (similarity, 4-DOF) fitting with iterative peeling.

    Validates: inlier count, determinant, convexity, aspect ratio, area.
    Strips inliers and repeats to find multiple instances in one cluster.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def peel(
        self,
        bid: str,
        matches: list,
        tpl: dict,
        kp_scene: List[cv2.KeyPoint],
        scene_shape: Tuple[int, ...],
    ) -> List[dict]:
        found: List[dict] = []
        pool = list(matches)
        while len(pool) >= self._cfg.min_inliers:
            result = self._fit(bid, pool, tpl, kp_scene, scene_shape)
            if result is None:
                break
            found.append(result)
            pool = self._strip(pool, tpl, kp_scene, result["H"])
        return found

    # ── private ──────────────────────────────────────────────────────────
    def _fit(self, bid, matches, tpl, kp_scene, scene_shape):
        src = np.float32([tpl["kp"][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_scene[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=self._cfg.ransac_thresh,
        )
        if H is None or mask is None:
            return None
        n_inliers = int(mask.ravel().sum())
        if n_inliers < self._cfg.min_inliers:
            return None

        det = np.linalg.det(H[:2, :2])
        if det < self._cfg.min_det or det > self._cfg.max_det:
            return None

        h, w = tpl["shape"][:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        bbox = cv2.transform(corners, H)
        if not cv2.isContourConvex(bbox.astype(np.int32)):
            return None

        pts4 = bbox.reshape(4, 2)
        w_det = (np.linalg.norm(pts4[1] - pts4[0]) + np.linalg.norm(pts4[2] - pts4[3])) / 2
        h_det = (np.linalg.norm(pts4[3] - pts4[0]) + np.linalg.norm(pts4[2] - pts4[1])) / 2
        if w_det < 1 or h_det < 1:
            return None
        ar_det = max(w_det, h_det) / min(w_det, h_det)
        ar_mod = max(w, h) / max(min(w, h), 1)
        tol = self._cfg.ar_tolerance
        if ar_det < ar_mod / tol or ar_det > ar_mod * tol:
            return None

        area = cv2.contourArea(bbox)
        scene_area = scene_shape[0] * scene_shape[1]
        if area < self._cfg.min_area or area > scene_area * self._cfg.max_area_frac:
            return None

        return {"book_id": bid, "score": n_inliers, "bbox": bbox, "H": H}

    def _strip(self, matches, tpl, kp_scene, H):
        src = np.float32([tpl["kp"][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_scene[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        proj = cv2.transform(src, H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        return [m for m, e in zip(matches, err) if e >= self._cfg.ransac_thresh * 1.5]


# ══════════════════════════════════════════════════════════════════════════
#  Edge-Density Verifier
# ══════════════════════════════════════════════════════════════════════════
class EdgeDensityVerifier:
    """Rejects detections whose Canny edge density in the warped ROI
    differs too much from the model's edge density."""

    def __init__(self, cfg: Config) -> None:
        self._tol = cfg.edge_tolerance
        self._templates: Dict[str, float] = {}

    def register(self, bid: str, enhanced: np.ndarray) -> None:
        edges = cv2.Canny(enhanced, 50, 150)
        self._templates[bid] = np.mean(edges > 0)

    def verify(self, det: dict, scene_enhanced: np.ndarray) -> bool:
        bid = det["book_id"]
        model_density = self._templates.get(bid)
        if model_density is None or model_density < 1e-6:
            return True
        mask = np.zeros(scene_enhanced.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, det["bbox"].astype(np.int32), 255)
        roi_pixels = np.sum(mask > 0)
        if roi_pixels == 0:
            return True
        scene_edges = cv2.Canny(scene_enhanced, 50, 150)
        scene_density = np.sum((scene_edges > 0) & (mask > 0)) / roi_pixels
        ratio = scene_density / model_density
        return (1.0 / self._tol) <= ratio <= self._tol


# ══════════════════════════════════════════════════════════════════════════
#  NCC Warp-and-Compare Verifier
# ══════════════════════════════════════════════════════════════════════════
class NCCVerifier:
    """Warps the model image into scene space using the estimated affine
    transform and computes Normalised Cross-Correlation.

    True positives have NCC ≈ 0.8, false positives ≈ 0.07.
    A threshold of 0.20 removes 95 % of FPs while losing < 2 % of TPs.
    """

    def __init__(self, cfg: Config) -> None:
        self._threshold = cfg.ncc_threshold
        self._models: Dict[str, np.ndarray] = {}

    def register(self, bid: str, enhanced: np.ndarray) -> None:
        self._models[bid] = enhanced

    def verify(self, det: dict, scene_enhanced: np.ndarray) -> Tuple[bool, float]:
        """Return (accepted, ncc_score)."""
        bid = det["book_id"]
        model_img = self._models.get(bid)
        if model_img is None:
            return True, 1.0
        H = det["H"]
        h_s, w_s = scene_enhanced.shape[:2]

        warp_fn = cv2.warpAffine if H.shape[0] == 2 else cv2.warpPerspective
        warped = warp_fn(model_img, H, (w_s, h_s),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = warp_fn(np.ones_like(model_img) * 255, H, (w_s, h_s),
                       flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        valid = mask > 128
        if valid.sum() < 100:
            return True, 1.0

        w_pix = warped[valid].astype(np.float64)
        s_pix = scene_enhanced[valid].astype(np.float64)
        w_pix -= w_pix.mean()
        s_pix -= s_pix.mean()
        w_std, s_std = w_pix.std(), s_pix.std()
        if w_std < 1e-8 or s_std < 1e-8:
            return False, 0.0
        ncc = float(np.dot(w_pix, s_pix) / (w_std * s_std * len(w_pix)))
        return ncc >= self._threshold, ncc


# ══════════════════════════════════════════════════════════════════════════
#  Non-Maximum Suppression
# ══════════════════════════════════════════════════════════════════════════
def _iou(p1: np.ndarray, p2: np.ndarray) -> float:
    ok, inter = cv2.intersectConvexConvex(p1, p2)
    if not ok or inter is None:
        return 0.0
    ia = cv2.contourArea(inter)
    return ia / (cv2.contourArea(p1) + cv2.contourArea(p2) - ia + 1e-7)


def nms(cands: List[dict], cfg: Config) -> List[dict]:
    """Score-ordered greedy NMS with two-tier IoU (stricter for same model)."""
    if not cands:
        return []
    cands.sort(key=lambda c: c["score"], reverse=True)
    kept: List[dict] = []
    for c in cands:
        cp = c["bbox"].reshape(4, 2).astype(np.float32)
        dup = False
        for k in kept:
            kp_ = k["bbox"].reshape(4, 2).astype(np.float32)
            thresh = cfg.iou_same if c["book_id"] == k["book_id"] else cfg.iou_cross
            if _iou(cp, kp_) > thresh:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


# ══════════════════════════════════════════════════════════════════════════
#  Visualizer  — saves per-scene annotated images + summary grid
# ══════════════════════════════════════════════════════════════════════════

# Fixed palette — 22 visually distinct colours (one per model)
MODEL_COLORS = {
    "model_0":  (230,  25,  75),  "model_1":  ( 60, 180,  75),
    "model_2":  (255, 225,  25),  "model_3":  (  0, 130, 200),
    "model_4":  (245, 130,  48),  "model_5":  (145,  30, 180),
    "model_6":  ( 70, 240, 240),  "model_7":  (240,  50, 230),
    "model_8":  (210, 245,  60),  "model_9":  (250, 190, 212),
    "model_10": (  0, 128, 128),  "model_11": (220, 190, 255),
    "model_12": (170, 110,  40),  "model_13": (255, 250, 200),
    "model_14": (128,   0,   0),  "model_15": (170, 255, 195),
    "model_16": (128, 128,   0),  "model_17": (255, 215, 180),
    "model_18": (  0,   0, 128),  "model_19": (128, 128, 128),
    "model_20": ( 60,  20,  90),  "model_21": ( 10, 190, 100),
}


def _color_for(bid: str) -> Tuple[int, int, int]:
    """Return a BGR colour for a book_id (persistent across scenes)."""
    rgb = MODEL_COLORS.get(bid, (200, 200, 200))
    return (rgb[2], rgb[1], rgb[0])          # RGB → BGR


def draw_detections(
    img: np.ndarray,
    results: List[dict],
    scene_name: str,
    gt: List[str],
    *,
    ncc_scores: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Draw bounding boxes + labels on a scene image.  Returns the annotated copy."""
    vis = img.copy()

    for i, r in enumerate(results):
        bbox = np.int32(r["bbox"])
        col = _color_for(r["book_id"])

        # semi-transparent fill
        overlay = vis.copy()
        cv2.fillConvexPoly(overlay, bbox, col)
        cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)

        # border
        cv2.polylines(vis, [bbox], True, col, 3, cv2.LINE_AA)

        # label
        ncc_str = ""
        if ncc_scores and i in ncc_scores:
            ncc_str = f"  NCC={ncc_scores[i]:.2f}"
        label = f"{r['book_id']} ({r['score']}){ncc_str}"
        fs, th = 0.50, 2
        (tw, thr), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        pts = bbox.reshape(-1, 2)
        x, y = pts[pts[:, 1].argmin()]
        x = max(0, min(x, vis.shape[1] - tw))
        y = max(thr + 12, y)
        cv2.rectangle(vis, (x, y - thr - 8), (x + tw + 4, y + 2), col, -1)
        cv2.putText(vis, label, (x + 2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

    # header bar with GT comparison
    detected = [r["book_id"] for r in results]
    gt_rem = list(gt)
    tp = 0
    for d in detected:
        if d in gt_rem:
            tp += 1
            gt_rem.remove(d)
    fp = len(detected) - tp
    fn = len(gt) - tp
    status = "PERFECT" if fp == 0 and fn == 0 else f"TP={tp} FP={fp} FN={fn}"

    header_h = 36
    header = np.zeros((header_h, vis.shape[1], 3), dtype=np.uint8)
    header[:] = (40, 40, 40)
    text = f"{scene_name}  |  Det={len(results)}  GT={len(gt)}  |  {status}"
    bar_col = (0, 200, 0) if (fp == 0 and fn == 0) else (0, 140, 255)
    cv2.rectangle(header, (0, 0), (vis.shape[1], header_h), bar_col, 3)
    cv2.putText(header, text, (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    vis = np.vstack([header, vis])
    return vis


# ══════════════════════════════════════════════════════════════════════════
#  Book Recognizer  (orchestrator)
# ══════════════════════════════════════════════════════════════════════════
class BookRecognizer:
    """Wires all pipeline stages together."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.prep      = Preprocessor(self.cfg)
        self.extractor = FeatureExtractor(self.cfg)
        self.matcher   = FeatureMatcher(self.cfg)
        self.voter     = StarVoter()
        self.clusterer = Clusterer(self.cfg)
        self.verifier  = GeometricVerifier(self.cfg)
        self.edge_vfy  = EdgeDensityVerifier(self.cfg) if self.cfg.edge_verify else None
        self.ncc_vfy   = NCCVerifier(self.cfg) if self.cfg.ncc_verify else None

    # ── model indexing ───────────────────────────────────────────────────
    def load_library(self, path: Path) -> None:
        print(f"Indexing models from {path} ...")
        n = 0
        for f in sorted(os.listdir(path)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(str(path / f))
            if img is None:
                continue
            bid = os.path.splitext(f)[0]
            enhanced = self.prep(img)
            kp, des = self.extractor(enhanced)
            if des is None:
                continue
            self.voter.register(bid, kp, enhanced.shape)
            self.matcher.add(bid, des)
            if self.edge_vfy:
                self.edge_vfy.register(bid, enhanced)
            if self.ncc_vfy:
                self.ncc_vfy.register(bid, enhanced)
            n += 1
            print(f"  {bid}: {len(kp):5d} keypoints")
        self.matcher.build()
        print(f"{n} models indexed.\n")

    # ── single-scene detection ───────────────────────────────────────────
    def detect(
        self,
        kp_scene: List[cv2.KeyPoint],
        des_scene: Optional[np.ndarray],
        scene_shape: Tuple[int, ...],
    ) -> List[dict]:
        if des_scene is None:
            return []
        model_kps = {bid: tpl["kp"] for bid, tpl in self.voter.templates.items()}
        cands: List[dict] = []
        for bid, matches in self.matcher.match(des_scene, kp_scene, model_kps).items():
            tpl = self.voter.templates[bid]
            if len(matches) < self.cfg.dbscan_min_pts:
                continue
            centres = self.voter.project(matches, kp_scene, tpl)
            clusters = self.clusterer(centres)
            for idxs in clusters.values():
                if len(idxs) < self.cfg.min_inliers:
                    continue
                cluster_matches = [matches[i] for i in idxs]
                cands.extend(self.verifier.peel(bid, cluster_matches, tpl, kp_scene, scene_shape))
        return nms(cands, self.cfg)

    # ── full run with visualisation ──────────────────────────────────────
    def run(self, scenes_path: Path, output_path: Path) -> Dict[str, List[dict]]:
        output_path.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f for f in os.listdir(scenes_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        print(f"Processing {len(files)} scenes ...\n")

        all_results: Dict[str, List[dict]] = {}
        annotated_images: List[np.ndarray] = []
        total = 0

        for f in files:
            scene_name = os.path.splitext(f)[0]
            img = cv2.imread(str(scenes_path / f))
            if img is None:
                continue

            enhanced = self.prep(img)
            kp, des = self.extractor(enhanced)
            results = self.detect(kp, des, enhanced.shape)

            # ── post-verification ────────────────────────────────────────
            if self.edge_vfy:
                results = [r for r in results if self.edge_vfy.verify(r, enhanced)]

            ncc_scores: Dict[int, float] = {}
            if self.ncc_vfy:
                kept = []
                for i, r in enumerate(results):
                    ok, score = self.ncc_vfy.verify(r, enhanced)
                    if ok:
                        ncc_scores[len(kept)] = score
                        kept.append(r)
                results = kept

            all_results[f] = results
            total += len(results)

            # ── draw & save ──────────────────────────────────────────────
            gt = GROUND_TRUTH.get(scene_name, [])
            vis = draw_detections(img, results, scene_name, gt, ncc_scores=ncc_scores)
            out_file = output_path / f"{scene_name}_result.jpg"
            cv2.imwrite(str(out_file), vis)
            annotated_images.append((scene_name, vis))

            det_ids = [r["book_id"] for r in results]
            mark = "OK" if self._check_scene(det_ids, gt) else "!!"
            print(f"  [{mark}] {scene_name}: detected {len(results)} → {det_ids}")

        print(f"\nTotal: {total} books detected across {len(files)} scenes")
        print(f"Annotated images saved to {output_path}/\n")

        # ── summary grid ─────────────────────────────────────────────────
        self._save_summary_grid(annotated_images, output_path)
        return all_results

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _check_scene(detected: List[str], gt: List[str]) -> bool:
        gt_rem = list(gt)
        tp = 0
        for d in detected:
            if d in gt_rem:
                tp += 1
                gt_rem.remove(d)
        return (len(detected) - tp) == 0 and (len(gt) - tp) == 0

    @staticmethod
    def _save_summary_grid(images: list, output_path: Path) -> None:
        """Save a large grid image with all annotated scenes."""
        if not images:
            return
        # Determine grid size
        n = len(images)
        cols = min(5, n)
        rows = (n + cols - 1) // cols

        # Resize all to a common thumbnail size
        thumb_w, thumb_h = 640, 400
        thumbs = []
        for name, img in images:
            resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            thumbs.append(resized)

        # Pad with black thumbnails if needed
        while len(thumbs) < rows * cols:
            thumbs.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))

        # Assemble grid
        grid_rows = []
        for r in range(rows):
            row_imgs = thumbs[r * cols: (r + 1) * cols]
            grid_rows.append(np.hstack(row_imgs))
        grid = np.vstack(grid_rows)

        cv2.imwrite(str(output_path / "summary_grid.jpg"), grid)
        print(f"Summary grid saved to {output_path / 'summary_grid.jpg'}")


# ══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════
def evaluate(all_results: Dict[str, List[dict]]) -> Dict[str, float]:
    """Compute precision / recall / F1 against ground truth."""
    total_tp = total_fp = total_fn = 0
    print(f"\n{'='*70}")
    print("  EVALUATION vs GROUND TRUTH")
    print(f"{'='*70}")

    for fname, results in sorted(all_results.items()):
        scene_key = os.path.splitext(fname)[0]
        gt = GROUND_TRUTH.get(scene_key, [])
        detected = [r["book_id"] for r in results]

        gt_rem = list(gt)
        tp = 0
        for d in detected:
            if d in gt_rem:
                tp += 1
                gt_rem.remove(d)
        fp = len(detected) - tp
        fn = len(gt) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if gt or detected:
            s = "  OK " if fp == 0 and fn == 0 else "  !! "
            print(f"{s} {scene_key}: GT={len(gt)}, Det={len(detected)}, TP={tp}, FP={fp}, FN={fn}")
            if fp > 0 or fn > 0:
                print(f"       GT:  {gt}")
                print(f"       Det: {detected}")

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print(f"\n{'─'*70}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print(f"{'='*70}\n")
    return {"precision": prec, "recall": rec, "f1": f1}


# ══════════════════════════════════════════════════════════════════════════
#  Display summary in matplotlib
# ══════════════════════════════════════════════════════════════════════════
def show_summary(output_path: Path) -> None:
    """Open the saved summary grid with matplotlib."""
    grid_file = output_path / "summary_grid.jpg"
    if not grid_file.exists():
        print("No summary grid found.")
        return
    grid = cv2.imread(str(grid_file))
    plt.figure(figsize=(22, 14))
    plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    plt.title("Book Recognition — All Scenes", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path / "summary_grid_figure.png"), dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path / 'summary_grid_figure.png'}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = Config()
    rec = BookRecognizer(cfg)
    rec.load_library(MODELS_DIR)

    all_results = rec.run(SCENES_DIR, OUTPUT_DIR)
    metrics = evaluate(all_results)

    # show interactive summary
    show_summary(OUTPUT_DIR)
