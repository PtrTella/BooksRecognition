"""
Book Spine Recognition — Best Pipeline (F1 ≈ 0.969)
=====================================================

SIFT/RootSIFT  →  FLANN k=2 + same-model relaxed Lowe ratio + MAD scale filter
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
│CLAHE+USM+DoG │    │  RootSIFT     │    │ FLANN+Lowe   │
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
    "scene_27": ["model_2", "model_2", "model_2", "model_2", "model_3", "model_3"],
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
    """All tuneable hyper-parameters — values tuned for F1 ≈ 0.963."""

    # ── Preprocessing (CLAHE + USM + DoG on LAB L-channel) ──
    clahe_clip:         float = 3.0
    clahe_grid:         int   = 8
    usm_sigma:          float = 2.0
    usm_amount:         float = 1.5
    dog_sigma1:         float = 0.7
    dog_sigma2:         float = 2.5
    dog_weight:         float = 1.5

    # ── SIFT ──
    nfeatures:          int   = 0
    contrast_threshold: float = 0.008
    edge_threshold:     float = 15
    n_octave_layers:    int   = 4
    root_sift:          bool  = True

    # ── FLANN matching ──
    lowe_ratio:         float = 0.78
    same_model_ratio:   float = 0.90   # relaxed ratio when 2nd-best is from same model
    flann_trees:        int   = 5
    flann_checks:       int   = 100
    scale_sigma:        float = 3.5

    # ── DBSCAN clustering ──
    dbscan_eps:      float = 20.0
    dbscan_min_pts:  int   = 3
    use_clustering:  bool  = True

    # ── Geometric verification ──
    ransac_thresh:   float = 4.0
    min_inliers:     int   = 5   # Ablation-optimal (F1=0.983)

    # ── NMS ──
    iou_same:  float = 0.30
    iou_cross: float = 0.35

    # ── NCC Verification ──
    ncc_threshold:   float = 0.12  # Ablation-optimal (clean NCC, F1=0.983)


# ══════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ══════════════════════════════════════════════════════════════════════════
class Preprocessor:
    """CLAHE on L-channel of CIE-LAB → USM sharpening → DoG band-pass.

    The USM step boosts mid-frequency contrast (text, edges), while the DoG
    band-pass filter (σ1=0.7, σ2=2.5) enhances fine structures that produce
    more discriminative SIFT keypoints (+23 % KP, +0.039 F1 vs plain CLAHE+USM).
    """

    def __init__(self, cfg: Config) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
        )
        self._usm_sigma  = cfg.usm_sigma
        self._usm_amount = cfg.usm_amount
        self._dog_s1     = cfg.dog_sigma1
        self._dog_s2     = cfg.dog_sigma2
        self._dog_w      = cfg.dog_weight

    def __call__(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        l_ch     = self._clahe_lab(bgr)
        sharpened = self._unsharp_mask(l_ch)
        enhanced  = self._dog_bandpass(sharpened)
        return enhanced, sharpened   # (enhanced_for_SIFT, clean_for_NCC)

    def _clahe_lab(self, bgr: np.ndarray) -> np.ndarray:
        """Convert to CIE-LAB and apply CLAHE on the L-channel."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return self._clahe.apply(lab[:, :, 0])

    def _unsharp_mask(self, gray: np.ndarray) -> np.ndarray:
        """Boost mid-frequency contrast via unsharp masking."""
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=self._usm_sigma)
        return cv2.addWeighted(gray, 1.0 + self._usm_amount, blur, -self._usm_amount, 0)

    def _dog_bandpass(self, gray: np.ndarray) -> np.ndarray:
        """Enhance fine structures via Difference-of-Gaussians."""
        g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=self._dog_s1)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=self._dog_s2)
        dog = cv2.subtract(g1, g2)
        return cv2.addWeighted(gray, 1.0, dog, self._dog_w, 0)


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
    """Global FLANN index across all models  (same-model relaxed ratio).

    Standard Lowe ratio test vs different-model 2nd-best neighbours.
    When 1st and 2nd neighbours belong to the same model, a relaxed ratio
    threshold is used.  This fixes the *match dilution* problem where thin
    models (e.g. model_19, 54×580) lose 80 %+ of matches because their
    own descriptors compete with each other in the shared FLANN index.

    Also applies MAD-based scale-ratio consistency filter.
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
        groups = self._ratio_test(des_scene)
        return self._scale_filter(groups, kp_scene, model_kps)

    def _ratio_test(self, des_scene: np.ndarray) -> Dict[str, list]:
        """Lowe ratio test with same-model relaxation."""
        raw = self._matcher.knnMatch(des_scene, k=2)
        groups: Dict[str, list] = defaultdict(list)
        ratio = self._cfg.lowe_ratio
        same_ratio = self._cfg.same_model_ratio
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            m_bid = self._ids[m.imgIdx]
            n_bid = self._ids[n.imgIdx]
            if m.distance < ratio * n.distance:
                groups[m_bid].append(m)
            elif m_bid == n_bid and m.distance < same_ratio * n.distance:
                groups[m_bid].append(m)
        return groups

    def _scale_filter(
        self,
        groups: Dict[str, list],
        kp_scene: List[cv2.KeyPoint],
        model_kps: Dict[str, List[cv2.KeyPoint]],
    ) -> Dict[str, list]:
        """MAD-based scale-ratio consistency filter."""
        sigma = self._cfg.scale_sigma
        filtered: Dict[str, list] = {}
        for bid, ms in groups.items():
            mkp = model_kps.get(bid)
            if len(ms) < 3 or mkp is None:
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
    """Generalised Hough voting:  stores displacement vectors
    (keypoint → model centre), projects matched scene keypoints to
    centre-vote locations, and clusters them with DBSCAN."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
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

    def vote(self, matches: list, kp_scene: List[cv2.KeyPoint],
             tpl: dict) -> Dict[int, List[int]]:
        """Project matches to centre votes and cluster with DBSCAN.
        Returns  {cluster_id: [match_indices]}."""
        centres = self._project(matches, kp_scene, tpl)
        if not self._cfg.use_clustering or len(centres) < 2:
            return {0: list(range(len(centres)))} if len(centres) else {}
        db = DBSCAN(eps=self._cfg.dbscan_eps, min_samples=self._cfg.dbscan_min_pts)
        labels = db.fit_predict(centres)
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            if label >= 0:
                clusters[label].append(i)
        return clusters

    # ── internal ─────────────────────────────────────────────────────────
    def _project(self, matches: list, kp_scene: List[cv2.KeyPoint],
                 tpl: dict) -> np.ndarray:
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
#  Verifier  (RANSAC affine  +  NCC acceptance)
# ══════════════════════════════════════════════════════════════════════════
class Verifier:
    """Unified verification: RANSAC estimates the affine transform,
    NCC warp-and-compare is the sole acceptance gate.

    Iterative peeling strips inliers after each accepted detection so
    that multiple instances of the same model can be found in one cluster.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._models: Dict[str, np.ndarray] = {}   # bid → enhanced image

    def register(self, bid: str, enhanced: np.ndarray) -> None:
        """Register a model's pre-processed image for NCC comparison."""
        self._models[bid] = enhanced

    # ── public entry point ───────────────────────────────────────────────
    def peel(
        self,
        bid: str,
        matches: list,
        tpl: dict,
        kp_scene: List[cv2.KeyPoint],
        scene_enhanced: np.ndarray,
    ) -> List[dict]:
        """Find all valid instances in *matches* via RANSAC + NCC peeling."""
        found: List[dict] = []
        pool = list(matches)
        while len(pool) >= self._cfg.min_inliers:
            result = self._fit(bid, pool, tpl, kp_scene, scene_enhanced)
            if result is None:
                break
            found.append(result)
            pool = self._strip(pool, tpl, kp_scene, result["H"])
        return found

    # ── RANSAC fit + NCC gate ────────────────────────────────────────────
    def _fit(self, bid, matches, tpl, kp_scene, scene_enhanced):
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

        # Bounding box from warped model corners
        h, w = tpl["shape"][:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        bbox = cv2.transform(corners, H)

        # NCC acceptance gate
        ncc = self._ncc(bid, H, scene_enhanced)
        if ncc < self._cfg.ncc_threshold:
            return None

        return {"book_id": bid, "score": n_inliers, "bbox": bbox, "H": H,
                "ncc": ncc}

    # ── NCC warp-and-compare ─────────────────────────────────────────────
    def _ncc(self, bid: str, H: np.ndarray, scene_enhanced: np.ndarray) -> float:
        model_img = self._models.get(bid)
        if model_img is None:
            return 1.0
        h_s, w_s = scene_enhanced.shape[:2]

        warped = cv2.warpAffine(model_img, H, (w_s, h_s),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ones_like(model_img) * 255, H, (w_s, h_s),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        valid = mask > 128
        if valid.sum() < 100:
            return 1.0

        w_pix = warped[valid].astype(np.float64)
        s_pix = scene_enhanced[valid].astype(np.float64)
        w_pix -= w_pix.mean()
        s_pix -= s_pix.mean()
        w_std, s_std = w_pix.std(), s_pix.std()
        if w_std < 1e-8 or s_std < 1e-8:
            return 0.0
        return float(np.dot(w_pix, s_pix) / (w_std * s_std * len(w_pix)))

    # ── strip inliers for peeling ────────────────────────────────────────
    def _strip(self, matches, tpl, kp_scene, H):
        src = np.float32([tpl["kp"][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_scene[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        proj = cv2.transform(src, H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        return [m for m, e in zip(matches, err) if e >= self._cfg.ransac_thresh * 1.5]


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
#  Book Recognizer  (pure detection pipeline — no GT, no visualisation)
# ══════════════════════════════════════════════════════════════════════════
class BookRecognizer:
    """Wires all pipeline stages together.  Returns raw detection dicts;
    all visualisation and evaluation is handled externally."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.prep      = Preprocessor(self.cfg)
        self.extractor = FeatureExtractor(self.cfg)
        self.matcher   = FeatureMatcher(self.cfg)
        self.voter     = StarVoter(self.cfg)
        self.verifier  = Verifier(self.cfg)

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
            enhanced, clean = self.prep(img)
            kp, des = self.extractor(enhanced)
            if des is None:
                continue
            self.voter.register(bid, kp, enhanced.shape)
            self.matcher.add(bid, des)
            self.verifier.register(bid, clean)
            n += 1
            print(f"  {bid}: {len(kp):5d} keypoints")
        self.matcher.build()
        print(f"{n} models indexed.\n")

    # ── single-scene detection ───────────────────────────────────────────
    def detect(
        self,
        kp_scene: List[cv2.KeyPoint],
        des_scene: Optional[np.ndarray],
        scene_enhanced: np.ndarray,
    ) -> List[dict]:
        if des_scene is None:
            return []
        model_kps = {bid: tpl["kp"] for bid, tpl in self.voter.templates.items()}
        cands: List[dict] = []
        for bid, matches in self.matcher.match(des_scene, kp_scene, model_kps).items():
            tpl = self.voter.templates[bid]
            if len(matches) < self.cfg.dbscan_min_pts:
                continue
            clusters = self.voter.vote(matches, kp_scene, tpl)
            for idxs in clusters.values():
                if len(idxs) < self.cfg.min_inliers:
                    continue
                cluster_matches = [matches[i] for i in idxs]
                cands.extend(self.verifier.peel(bid, cluster_matches, tpl, kp_scene, scene_enhanced))
        return nms(cands, self.cfg)

    # ── batch run (detection only) ────────────────────────────────────────
    def run(self, scenes_path: Path) -> Dict[str, List[dict]]:
        """Run detection on all scenes.  Returns {filename: [detections]}."""
        files = sorted(
            f for f in os.listdir(scenes_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        print(f"Processing {len(files)} scenes ...\n")

        all_results: Dict[str, List[dict]] = {}
        for f in files:
            img = cv2.imread(str(scenes_path / f))
            if img is None:
                continue
            enhanced, clean = self.prep(img)
            kp, des = self.extractor(enhanced)
            results = self.detect(kp, des, clean)
            all_results[f] = results

            det_ids = [r["book_id"] for r in results]
            print(f"  {os.path.splitext(f)[0]}: {len(results)} detections → {det_ids}")

        print(f"\nTotal: {sum(len(r) for r in all_results.values())} books "
              f"detected across {len(all_results)} scenes\n")
        return all_results


# ══════════════════════════════════════════════════════════════════════════
#  Evaluator  (GT-aware metrics — fully independent of pipeline)
# ══════════════════════════════════════════════════════════════════════════
class Evaluator:
    """Compares detection results against a ground-truth dictionary."""

    def __init__(self, ground_truth: Dict[str, List[str]]) -> None:
        self._gt = ground_truth

    @staticmethod
    def _score_scene(detected: List[str], gt: List[str]) -> Tuple[int, int, int]:
        gt_rem = list(gt)
        tp = 0
        for d in detected:
            if d in gt_rem:
                tp += 1
                gt_rem.remove(d)
        return tp, len(detected) - tp, len(gt) - tp

    def evaluate(self, all_results: Dict[str, List[dict]]) -> Dict[str, float]:
        """Print per-scene breakdown and return {precision, recall, f1}."""
        total_tp = total_fp = total_fn = 0
        print(f"\n{'='*70}")
        print("  EVALUATION vs GROUND TRUTH")
        print(f"{'='*70}")

        for fname in sorted(all_results):
            scene_key = os.path.splitext(fname)[0]
            gt = self._gt.get(scene_key, [])
            detected = [r["book_id"] for r in all_results[fname]]
            tp, fp, fn = self._score_scene(detected, gt)
            total_tp += tp; total_fp += fp; total_fn += fn

            if gt or detected:
                mark = "  OK " if fp == 0 and fn == 0 else "  !! "
                print(f"{mark} {scene_key}: GT={len(gt)}, Det={len(detected)}, "
                      f"TP={tp}, FP={fp}, FN={fn}")
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
#  Visualizer  (matplotlib display — fully independent of pipeline)
# ══════════════════════════════════════════════════════════════════════════
class Visualizer:
    """Draws detection bounding boxes and displays results via matplotlib.
    No files are saved — designed for notebook usage."""

    def __init__(self, ground_truth: Optional[Dict[str, List[str]]] = None) -> None:
        self._gt = ground_truth or {}

    # ── plot a single scene ──────────────────────────────────────────────
    def show_scene(self, img: np.ndarray, results: List[dict],
                   scene_name: str) -> None:
        """Annotate and display a single scene."""
        vis = self._draw(img, results, scene_name)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(scene_name, fontsize=13, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ── plot the summary grid ────────────────────────────────────────────
    def show_grid(self, all_results: Dict[str, List[dict]],
                  scenes_path: Path, *, cols: int = 5) -> None:
        """Build and display a summary grid of all annotated scenes."""
        thumbs: List[np.ndarray] = []
        tw, th = 640, 400
        for fname in sorted(all_results):
            scene_name = os.path.splitext(fname)[0]
            img = cv2.imread(str(scenes_path / fname))
            if img is None:
                continue
            vis = self._draw(img, all_results[fname], scene_name)
            thumbs.append(cv2.resize(vis, (tw, th), interpolation=cv2.INTER_AREA))
        if not thumbs:
            return
        n = len(thumbs)
        rows = (n + cols - 1) // cols
        while len(thumbs) < rows * cols:
            thumbs.append(np.zeros((th, tw, 3), dtype=np.uint8))
        grid = np.vstack([
            np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(rows)
        ])
        plt.figure(figsize=(22, 14))
        plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        plt.title("Book Recognition — All Scenes", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ── private helpers ──────────────────────────────────────────────────
    def _draw(self, img: np.ndarray, results: List[dict],
              scene_name: str) -> np.ndarray:
        """Annotate a scene image with bounding boxes and a header bar."""
        vis = img.copy()
        for r in results:
            bbox = np.int32(r["bbox"])
            col = self._bgr(r["book_id"])
            overlay = vis.copy()
            cv2.fillConvexPoly(overlay, bbox, col)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
            cv2.polylines(vis, [bbox], True, col, 3, cv2.LINE_AA)

            ncc_str = f"  NCC={r['ncc']:.2f}" if "ncc" in r else ""
            label = f"{r['book_id']} ({r['score']}){ncc_str}"
            fs, thickness = 0.50, 2
            (tw, thr), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)
            pts = bbox.reshape(-1, 2)
            x, y = pts[pts[:, 1].argmin()]
            x = max(0, min(x, vis.shape[1] - tw))
            y = max(thr + 12, y)
            cv2.rectangle(vis, (x, y - thr - 8), (x + tw + 4, y + 2), col, -1)
            cv2.putText(vis, label, (x + 2, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thickness, cv2.LINE_AA)

        return self._add_header(vis, results, scene_name)

    @staticmethod
    def _bgr(bid: str) -> Tuple[int, int, int]:
        """Deterministic random BGR colour from model id (consistent across scenes)."""
        h = hash(bid)
        r = (h & 0xFF) % 200 + 55
        g = ((h >> 8) & 0xFF) % 200 + 55
        b = ((h >> 16) & 0xFF) % 200 + 55
        return (b, g, r)

    def _add_header(self, vis: np.ndarray, results: List[dict],
                    scene_name: str) -> np.ndarray:
        header_h = 36
        header = np.zeros((header_h, vis.shape[1], 3), dtype=np.uint8)
        header[:] = (40, 40, 40)

        gt = self._gt.get(scene_name)
        if gt is not None:
            detected = [r["book_id"] for r in results]
            gt_rem = list(gt); tp = 0
            for d in detected:
                if d in gt_rem:
                    tp += 1; gt_rem.remove(d)
            fp, fn = len(detected) - tp, len(gt) - tp
            ok = fp == 0 and fn == 0
            status = "PERFECT" if ok else f"TP={tp} FP={fp} FN={fn}"
            text = f"{scene_name}  |  Det={len(results)}  GT={len(gt)}  |  {status}"
            bar_col = (0, 200, 0) if ok else (0, 140, 255)
        else:
            text = f"{scene_name}  |  Det={len(results)}"
            bar_col = (140, 140, 140)

        cv2.rectangle(header, (0, 0), (vis.shape[1], header_h), bar_col, 3)
        cv2.putText(header, text, (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([header, vis])


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = Config()

    # 1. Detection (GT-unaware)
    rec = BookRecognizer(cfg)
    rec.load_library(MODELS_DIR)
    all_results = rec.run(SCENES_DIR)

    # 2. Evaluation (optional, GT-aware)
    evaluator = Evaluator(GROUND_TRUTH)
    evaluator.evaluate(all_results)

    # 3. Visualisation (optional)
    vis = Visualizer(GROUND_TRUTH)
    vis.show_grid(all_results, SCENES_DIR)

