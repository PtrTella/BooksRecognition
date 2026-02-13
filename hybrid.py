"""
Book Spine Recognition -- Hybrid SIFT + LightGlue Pipeline
===========================================================

Combines the best of both worlds:
  - SIFT keypoints with orientation & scale  → proper star-model voting
  - LightGlue learned matcher                → more discriminative matching
  - Classical orientation-aware GHT           → multi-instance separation
  - Hierarchical clustering + affine peeling  → geometric verification

This hybrid leverages SIFT's rich geometric info (orientation, scale) for
the star-model centre-vote projection while replacing FLANN ratio-test
matching with LightGlue's learned matching, which can be more selective.

Usage:
  python hybrid.py                 # default: sift+lightglue
  python hybrid.py sift_lg         # SIFT keypoints + LightGlue
  python hybrid.py superpoint      # SuperPoint + LightGlue
  python hybrid.py aliked          # ALIKED + LightGlue
  python hybrid.py ensemble        # SIFT-FLANN + SIFT-LightGlue ensemble
"""

import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fclusterdata

warnings.filterwarnings("ignore")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ═══════════════════════════════════════════════════════════════════════════
#  Device
# ═══════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[DEVICE] Using: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration  (best params from debug tuning)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Backend: "sift_lg", "superpoint", "aliked", "ensemble"
    backend: str = "sift_lg"

    # Preprocessing
    clahe_clip:     float = 3.0
    clahe_grid:     int   = 8
    sharpen_amount: float = 1.0

    # SIFT extraction
    nfeatures:          int   = 2048    # cap for LightGlue performance
    contrast_threshold: float = 0.01
    edge_threshold:     float = 15
    n_octave_layers:    int   = 4
    root_sift:          bool  = True    # LightGlue needs normalized descriptors

    # LightGlue matching
    match_threshold: float = 0.0    # LightGlue score threshold (0 = let LG decide)
    max_keypoints:   int   = 4096

    # FLANN matching (for ensemble / fallback)
    lowe_ratio:   float = 0.78
    flann_trees:  int   = 5
    flann_checks: int   = 100
    scale_sigma:  float = 3.5

    # GHT voting
    bin_size:     int   = 18
    merge_radius: int   = 0
    min_votes:    int   = 3

    # Hierarchical clustering
    cluster_dist: float = 36.0

    # Geometric verification
    ransac_thresh: float = 4.0
    min_inliers:   int   = 3
    min_inliers_h: int   = 5
    affine_thresh: int   = 10
    min_area:      int   = 450
    min_det:       float = 0.025
    max_det:       float = 40.0
    max_area_frac: float = 0.9

    # Adaptive score threshold
    score_threshold_ratio: float = 0.003

    # NMS
    iou_same:  float = 0.30
    iou_cross: float = 0.35

    # Output
    save_output: bool = True
    output_dir:  str  = "output_hybrid"


# ═══════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

class Preprocessor:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
        )

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        l = self._clahe.apply(l)
        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=2.0)
        alpha = self._cfg.sharpen_amount
        return cv2.addWeighted(l, 1.0 + alpha, blur, -alpha, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  Feature Backends
# ═══════════════════════════════════════════════════════════════════════════

class SIFTLightGlueBackend:
    """SIFT keypoints + descriptors, matched by LightGlue.

    Preserves SIFT's orientation & scale for star-model voting while
    using LightGlue's learned matching instead of FLANN ratio test.
    """

    def __init__(self, cfg: Config):
        from kornia.feature import LightGlue
        self.cfg = cfg
        self.sift = cv2.SIFT_create(
            nfeatures=cfg.nfeatures,
            contrastThreshold=cfg.contrast_threshold,
            edgeThreshold=cfg.edge_threshold,
            nOctaveLayers=cfg.n_octave_layers,
        )
        self.lg = LightGlue(features="sift").to(DEVICE).eval()

    def extract(self, gray: np.ndarray) -> dict:
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return {"kp": [], "pts": np.zeros((0, 2)), "desc": np.zeros((0, 128)),
                    "scales": np.array([]), "oris": np.array([]), "n": 0}
        des = des.astype(np.float32)
        if self.cfg.root_sift:
            des /= des.sum(axis=1, keepdims=True) + 1e-7
            np.sqrt(des, out=des)
        pts = np.array([k.pt for k in kp], dtype=np.float32)
        scales = np.array([k.size for k in kp], dtype=np.float32)
        oris = np.array([k.angle for k in kp], dtype=np.float32)
        return {"kp": kp, "pts": pts, "desc": des, "scales": scales,
                "oris": oris, "n": len(kp)}

    def match(self, feat_m: dict, feat_s: dict,
              shape_m: Tuple[int, int], shape_s: Tuple[int, int]) -> np.ndarray:
        if feat_m["n"] == 0 or feat_s["n"] == 0:
            return np.zeros((0, 2), dtype=int)

        with torch.no_grad():
            inp = {
                "image0": {
                    "keypoints": torch.from_numpy(feat_m["pts"]).float().unsqueeze(0).to(DEVICE),
                    "descriptors": torch.from_numpy(feat_m["desc"]).float().unsqueeze(0).to(DEVICE),
                    "image_size": torch.tensor([shape_m[1], shape_m[0]]).unsqueeze(0).to(DEVICE),
                    "oris": torch.from_numpy(feat_m["oris"]).float().unsqueeze(0).to(DEVICE),
                    "scales": torch.from_numpy(feat_m["scales"]).float().unsqueeze(0).to(DEVICE),
                },
                "image1": {
                    "keypoints": torch.from_numpy(feat_s["pts"]).float().unsqueeze(0).to(DEVICE),
                    "descriptors": torch.from_numpy(feat_s["desc"]).float().unsqueeze(0).to(DEVICE),
                    "image_size": torch.tensor([shape_s[1], shape_s[0]]).unsqueeze(0).to(DEVICE),
                    "oris": torch.from_numpy(feat_s["oris"]).float().unsqueeze(0).to(DEVICE),
                    "scales": torch.from_numpy(feat_s["scales"]).float().unsqueeze(0).to(DEVICE),
                },
            }
            result = self.lg(inp)

        m01 = result["matches"][0].cpu().numpy()
        scores = result["scores"][0].cpu().numpy()
        valid = (m01[:, 0] >= 0) & (m01[:, 1] >= 0)
        if self.cfg.match_threshold > 0:
            valid &= scores >= self.cfg.match_threshold
        if not valid.any():
            return np.zeros((0, 2), dtype=int)
        return m01[valid].astype(int)


class SuperPointBackend:
    """SuperPoint + LightGlue — learned keypoints with learned matcher."""

    def __init__(self, cfg: Config):
        from kornia.feature import LightGlue
        # Try to import SuperPoint from kornia
        try:
            from kornia.feature import SuperPoint
            self.sp = SuperPoint(max_num_keypoints=cfg.max_keypoints).to(DEVICE).eval()
        except ImportError:
            # Fallback: use kornia's KeyPointDetector
            raise ImportError("SuperPoint not available in this kornia version")
        self.cfg = cfg
        self.lg = LightGlue(features="superpoint").to(DEVICE).eval()

    def extract(self, gray: np.ndarray) -> dict:
        t = torch.from_numpy(gray).float() / 255.0
        t = t.unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = self.sp(t)
        kp = feats["keypoints"][0].cpu().numpy()
        desc = feats["descriptors"][0].cpu().numpy()
        if len(kp) == 0:
            return {"kp": [], "pts": np.zeros((0, 2)), "desc": np.zeros((0, 256)),
                    "scales": np.array([]), "oris": np.array([]), "n": 0}
        # SuperPoint doesn't provide orientation/scale — use dummy values
        scales = np.ones(len(kp), dtype=np.float32) * 8.0
        oris = np.zeros(len(kp), dtype=np.float32)
        return {"kp": [], "pts": kp.astype(np.float32), "desc": desc.astype(np.float32),
                "scales": scales, "oris": oris, "n": len(kp)}

    def match(self, feat_m: dict, feat_s: dict,
              shape_m: Tuple[int, int], shape_s: Tuple[int, int]) -> np.ndarray:
        if feat_m["n"] == 0 or feat_s["n"] == 0:
            return np.zeros((0, 2), dtype=int)
        with torch.no_grad():
            inp = {
                "image0": {
                    "keypoints": torch.from_numpy(feat_m["pts"]).float().unsqueeze(0).to(DEVICE),
                    "descriptors": torch.from_numpy(feat_m["desc"]).float().unsqueeze(0).to(DEVICE),
                    "image_size": torch.tensor([shape_m[1], shape_m[0]]).unsqueeze(0).to(DEVICE),
                },
                "image1": {
                    "keypoints": torch.from_numpy(feat_s["pts"]).float().unsqueeze(0).to(DEVICE),
                    "descriptors": torch.from_numpy(feat_s["desc"]).float().unsqueeze(0).to(DEVICE),
                    "image_size": torch.tensor([shape_s[1], shape_s[0]]).unsqueeze(0).to(DEVICE),
                },
            }
            result = self.lg(inp)
        m01 = result["matches"][0].cpu().numpy()
        scores = result["scores"][0].cpu().numpy()
        valid = (m01[:, 0] >= 0) & (m01[:, 1] >= 0)
        if self.cfg.match_threshold > 0:
            valid &= scores >= self.cfg.match_threshold
        if not valid.any():
            return np.zeros((0, 2), dtype=int)
        return m01[valid].astype(int)


class ALIKEDBackend:
    """ALIKED + LightGlue — another learned detector/descriptor."""

    def __init__(self, cfg: Config):
        from kornia.feature import ALIKED, LightGlue
        self.cfg = cfg
        self.aliked = ALIKED(max_num_keypoints=cfg.max_keypoints,
                             detection_threshold=0.01).to(DEVICE).eval()
        self.lg = LightGlue(features="aliked").to(DEVICE).eval()

    def extract(self, gray: np.ndarray) -> dict:
        t = torch.from_numpy(gray).float() / 255.0
        t = t.unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = self.aliked(t)
        kp = feats["keypoints"][0].cpu().numpy()
        desc = feats["descriptors"][0].cpu().numpy()
        if len(kp) == 0:
            return {"kp": [], "pts": np.zeros((0, 2)), "desc": np.zeros((0, 128)),
                    "scales": np.array([]), "oris": np.array([]), "n": 0}
        scales = np.ones(len(kp), dtype=np.float32) * 8.0
        oris = np.zeros(len(kp), dtype=np.float32)
        return {"kp": [], "pts": kp.astype(np.float32), "desc": desc.astype(np.float32),
                "scales": scales, "oris": oris, "n": len(kp)}

    def match(self, feat_m: dict, feat_s: dict,
              shape_m: Tuple[int, int], shape_s: Tuple[int, int]) -> np.ndarray:
        if feat_m["n"] == 0 or feat_s["n"] == 0:
            return np.zeros((0, 2), dtype=int)
        with torch.no_grad():
            inp = {
                "image0": {
                    "keypoints": torch.from_numpy(feat_m["pts"]).float().unsqueeze(0).to(DEVICE),
                    "descriptors": torch.from_numpy(feat_m["desc"]).float().unsqueeze(0).to(DEVICE),
                    "image_size": torch.tensor([shape_m[1], shape_m[0]]).unsqueeze(0).to(DEVICE),
                },
                "image1": {
                    "keypoints": torch.from_numpy(feat_s["pts"]).float().unsqueeze(0).to(DEVICE),
                    "descriptors": torch.from_numpy(feat_s["desc"]).float().unsqueeze(0).to(DEVICE),
                    "image_size": torch.tensor([shape_s[1], shape_s[0]]).unsqueeze(0).to(DEVICE),
                },
            }
            result = self.lg(inp)
        m01 = result["matches"][0].cpu().numpy()
        scores = result["scores"][0].cpu().numpy()
        valid = (m01[:, 0] >= 0) & (m01[:, 1] >= 0)
        if self.cfg.match_threshold > 0:
            valid &= scores >= self.cfg.match_threshold
        if not valid.any():
            return np.zeros((0, 2), dtype=int)
        return m01[valid].astype(int)


# ═══════════════════════════════════════════════════════════════════════════
#  Star-Model Voter (orientation-aware for SIFT keypoints)
# ═══════════════════════════════════════════════════════════════════════════

class StarVoter:
    """GHT centre-vote with derotation and scale normalisation.

    Uses SIFT keypoint orientation + scale for proper multi-instance
    separation. Falls back to position-only voting for backends without
    orientation/scale info.
    """

    def __init__(self):
        self._templates: Dict[str, dict] = {}

    @property
    def templates(self) -> Dict[str, dict]:
        return self._templates

    def register(self, bid: str, feat: dict, shape: Tuple[int, ...]):
        h, w = shape[:2]
        cx, cy = w / 2.0, h / 2.0
        pts = feat["pts"]
        oris = feat["oris"]
        scales = feat["scales"]
        has_ori = np.any(oris != 0)

        if has_ori:
            # Full orientation-aware voting (SIFT)
            angles = np.deg2rad(oris)
            d = np.array([cx, cy]) - pts
            cos_a, sin_a = np.cos(angles), np.sin(angles)
            vecs = np.column_stack([
                 cos_a * d[:, 0] + sin_a * d[:, 1],
                -sin_a * d[:, 0] + cos_a * d[:, 1],
            ]) / (scales[:, np.newaxis] + 1e-8)
        else:
            # Position-only fallback (SuperPoint, ALIKED)
            diag = np.sqrt(w**2 + h**2)
            d = np.array([cx, cy]) - pts
            vecs = d / (diag + 1e-8)

        self._templates[bid] = {
            "pts": pts, "oris": oris, "scales": scales,
            "vecs": vecs, "shape": shape, "has_ori": has_ori,
            "n_kp": len(pts),
        }

    def project(self, match_idxs: np.ndarray, feat_model: dict,
                feat_scene: dict, tpl: dict) -> np.ndarray:
        m_idx = match_idxs[:, 0]
        s_idx = match_idxs[:, 1]
        pts_s = feat_scene["pts"][s_idx]

        if tpl["has_ori"]:
            # Orientation & scale-aware projection
            oris_s = feat_scene["oris"][s_idx]
            scales_s = feat_scene["scales"][s_idx]
            angles = np.deg2rad(oris_s)
            stored = tpl["vecs"][m_idx]
            cos_a, sin_a = np.cos(angles), np.sin(angles)
            return pts_s + np.column_stack([
                cos_a * stored[:, 0] - sin_a * stored[:, 1],
                sin_a * stored[:, 0] + cos_a * stored[:, 1],
            ]) * scales_s[:, np.newaxis]
        else:
            # Position-only: use bounding-box scale estimate
            vecs = tpl["vecs"][m_idx]
            pts_m = feat_model["pts"][m_idx]
            if len(pts_s) > 1:
                bb_s = pts_s.max(0) - pts_s.min(0)
                bb_m = pts_m.max(0) - pts_m.min(0)
                spread_s = np.sqrt(bb_s[0]**2 + bb_s[1]**2) + 1e-8
                spread_m = np.sqrt(bb_m[0]**2 + bb_m[1]**2) + 1e-8
                scale = spread_s / spread_m
            else:
                scale = 1.0
            h, w = tpl["shape"][:2]
            diag = np.sqrt(w**2 + h**2)
            return pts_s + vecs * diag * scale


# ═══════════════════════════════════════════════════════════════════════════
#  Geometric Verifier (affine peeling with two-tier strategy)
# ═══════════════════════════════════════════════════════════════════════════

class GeometricVerifier:
    def __init__(self, cfg: Config):
        self._cfg = cfg

    def peel(self, bid: str, pts_model: np.ndarray, pts_scene: np.ndarray,
             match_idxs: np.ndarray, model_shape: Tuple[int, ...],
             scene_shape: Tuple[int, ...], n_model_kp: int) -> List[dict]:
        found = []
        pool = match_idxs.copy()
        while len(pool) >= self._cfg.min_inliers:
            r = self._fit(bid, pts_model, pts_scene, pool,
                         model_shape, scene_shape, n_model_kp)
            if r is None:
                break
            found.append(r)
            pool = self._strip(pool, pts_model, pts_scene, r["H"])
        return found

    def _fit(self, bid, pts_m, pts_s, midxs, m_shape, s_shape, n_mkp):
        src = pts_m[midxs[:, 0]].reshape(-1, 1, 2).astype(np.float32)
        dst = pts_s[midxs[:, 1]].reshape(-1, 1, 2).astype(np.float32)

        if len(midxs) >= self._cfg.affine_thresh:
            H, mask = cv2.estimateAffine2D(
                src, dst, method=cv2.RANSAC,
                ransacReprojThreshold=self._cfg.ransac_thresh)
        else:
            H, mask = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC,
                ransacReprojThreshold=self._cfg.ransac_thresh)

        if H is None or mask is None:
            return None

        n_in = int(mask.ravel().sum())
        if n_in < self._cfg.min_inliers:
            return None

        score_thr = max(3, int(n_mkp * self._cfg.score_threshold_ratio))
        if n_in < score_thr:
            return None

        det = np.linalg.det(H[:2, :2])
        if det < self._cfg.min_det or det > self._cfg.max_det:
            return None

        h, w = m_shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        bbox = cv2.transform(corners, H)

        if not cv2.isContourConvex(bbox.astype(np.int32)):
            return None

        area = cv2.contourArea(bbox)
        sa = s_shape[0] * s_shape[1]
        if area < self._cfg.min_area or area > sa * self._cfg.max_area_frac:
            return None

        return {"book_id": bid, "score": n_in, "bbox": bbox, "H": H}

    def _strip(self, midxs, pts_m, pts_s, H):
        src = pts_m[midxs[:, 0]].reshape(-1, 1, 2).astype(np.float32)
        dst = pts_s[midxs[:, 1]].astype(np.float32)
        proj = cv2.transform(src, H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        return midxs[err >= self._cfg.ransac_thresh * 1.5]


# ═══════════════════════════════════════════════════════════════════════════
#  NMS
# ═══════════════════════════════════════════════════════════════════════════

def nms(cands: List[dict], cfg: Config) -> List[dict]:
    if not cands:
        return []
    cands.sort(key=lambda c: c["score"], reverse=True)
    kept = []
    for c in cands:
        cp = c["bbox"].reshape(4, 2).astype(np.float32)
        dup = False
        for k in kept:
            kp_ = k["bbox"].reshape(4, 2).astype(np.float32)
            thr = cfg.iou_same if c["book_id"] == k["book_id"] else cfg.iou_cross
            if _iou(cp, kp_) > thr:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def _iou(p1, p2):
    ok, inter = cv2.intersectConvexConvex(p1, p2)
    if not ok or inter is None:
        return 0.0
    ia = cv2.contourArea(inter)
    return ia / (cv2.contourArea(p1) + cv2.contourArea(p2) - ia + 1e-7)


# ═══════════════════════════════════════════════════════════════════════════
#  Visualizer
# ═══════════════════════════════════════════════════════════════════════════

class Visualizer:
    @staticmethod
    def draw(img, results, name, cfg: Config):
        vis = img.copy()
        rng = np.random.RandomState(42)
        n = max(len(results), 1)
        colors = [tuple(int(c) for c in row)
                  for row in rng.randint(50, 230, (n, 3)).tolist()]
        ids = []
        for i, r in enumerate(results):
            bbox = np.int32(r["bbox"])
            col = colors[i % n]
            ids.append(r["book_id"])
            overlay = vis.copy()
            cv2.fillConvexPoly(overlay, bbox, col)
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            cv2.polylines(vis, [bbox], True, col, 3, cv2.LINE_AA)
            label = f"{r['book_id']} ({r['score']})"
            fs, th = 0.55, 2
            (tw, thr), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            pts = bbox.reshape(-1, 2)
            x, y = pts[pts[:, 1].argmin()]
            x = max(0, min(x, vis.shape[1] - tw))
            y = max(thr + 12, y)
            cv2.rectangle(vis, (x, y - thr - 8), (x + tw + 4, y + 2), col, -1)
            cv2.putText(vis, label, (x + 2, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        if cfg.save_output:
            os.makedirs(cfg.output_dir, exist_ok=True)
            out = os.path.join(cfg.output_dir, name)
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{name} -- {len(results)} book(s): {', '.join(ids)}", fontsize=10)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)
        for r in results:
            print(f"  -> {r['book_id']} ({r['score']} inliers)")


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class BookRecognizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prep = Preprocessor(cfg)
        self.voter = StarVoter()
        self.verifier = GeometricVerifier(cfg)
        self.viz = Visualizer()
        self.model_features: Dict[str, dict] = {}
        self.model_shapes: Dict[str, Tuple[int, ...]] = {}

        print(f"[BACKEND] Initializing {cfg.backend.upper()} ...")
        t0 = time.time()
        if cfg.backend == "sift_lg":
            self.backend = SIFTLightGlueBackend(cfg)
        elif cfg.backend == "superpoint":
            self.backend = SuperPointBackend(cfg)
        elif cfg.backend == "aliked":
            self.backend = ALIKEDBackend(cfg)
        else:
            raise ValueError(f"Unknown backend: {cfg.backend}")
        print(f"[BACKEND] {cfg.backend.upper()} ready in {time.time()-t0:.1f}s")

    def load_library(self, path: str):
        print(f"Indexing models from {path} ...")
        n = 0
        for f in sorted(os.listdir(path)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(path, f))
            if img is None:
                continue
            bid = os.path.splitext(f)[0]
            enhanced = self.prep(img)
            self.model_shapes[bid] = enhanced.shape

            feat = self.backend.extract(enhanced)
            self.model_features[bid] = feat
            if feat["n"] > 0:
                self.voter.register(bid, feat, enhanced.shape)
            n += 1
            print(f"  {bid}: {feat['n']} keypoints")
        print(f"{n} models indexed.\n")

    def detect(self, img: np.ndarray) -> List[dict]:
        enhanced = self.prep(img)
        scene_shape = enhanced.shape
        scene_feat = self.backend.extract(enhanced)

        if scene_feat["n"] == 0:
            return []

        cands = []
        for bid, model_feat in self.model_features.items():
            if model_feat["n"] == 0:
                continue
            tpl = self.voter.templates.get(bid)
            if tpl is None:
                continue

            match_idxs = self.backend.match(
                model_feat, scene_feat,
                self.model_shapes[bid][:2], scene_shape[:2],
            )
            if len(match_idxs) < self.cfg.min_votes:
                continue

            print(f"    {bid}: {len(match_idxs)} matches", end="")

            # Centre votes
            centres = self.voter.project(
                match_idxs, model_feat, scene_feat, tpl,
            )

            # Hierarchical clustering
            if len(centres) >= 2:
                try:
                    labels = fclusterdata(
                        centres, t=self.cfg.cluster_dist,
                        criterion="distance", method="complete",
                    )
                except Exception:
                    labels = np.ones(len(centres), dtype=int)
            else:
                labels = np.ones(len(centres), dtype=int)

            clusters = defaultdict(list)
            for i, lab in enumerate(labels):
                clusters[lab].append(i)

            n_cl = len(clusters)
            for cl_idxs in clusters.values():
                if len(cl_idxs) < self.cfg.min_votes:
                    continue
                cl_matches = match_idxs[np.array(cl_idxs)]
                cands.extend(self.verifier.peel(
                    bid, model_feat["pts"], scene_feat["pts"],
                    cl_matches, self.model_shapes[bid],
                    scene_shape, tpl["n_kp"],
                ))

            n_f = sum(1 for c in cands if c["book_id"] == bid)
            print(f" → {n_f} det, {n_cl} cl")

        return nms(cands, self.cfg)

    def run_with_eval(self, scenes_path: str) -> Dict[str, List[dict]]:
        files = sorted(f for f in os.listdir(scenes_path)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
        print(f"Processing {len(files)} scenes ...")
        all_results = {}
        total = 0
        t_start = time.time()

        for f in files:
            img = cv2.imread(os.path.join(scenes_path, f))
            if img is None:
                continue
            print(f"\n--- {f} ---")
            t0 = time.time()
            results = self.detect(img)
            dt = time.time() - t0
            self.viz.draw(img, results, f, self.cfg)
            all_results[f] = results
            total += len(results)
            print(f"  [{dt:.1f}s] {len(results)} detections")

        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"TOTALE: {total} libri detectati su {len(files)} scene in {elapsed:.0f}s")
        print(f"{'='*60}")
        return all_results


# ═══════════════════════════════════════════════════════════════════════════
#  Ground Truth & Evaluation
# ═══════════════════════════════════════════════════════════════════════════

GROUND_TRUTH = {
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


def evaluate_results(all_results: Dict[str, List[dict]]) -> Dict[str, float]:
    total_tp = total_fp = total_fn = 0
    print(f"\n{'='*70}")
    print("VALUTAZIONE vs GROUND TRUTH")
    print(f"{'='*70}")

    for scene_name, results in sorted(all_results.items()):
        scene_key = scene_name.replace(".jpg", "").replace(".png", "")
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
            s = "✓" if fp == 0 and fn == 0 else "✗"
            print(f"  {s} {scene_key}: GT={len(gt)}, Det={len(detected)}, TP={tp}, FP={fp}, FN={fn}")
            if fp > 0 or fn > 0:
                print(f"      GT: {gt}")
                print(f"      Detected: {detected}")

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"\n{'='*70}")
    print(f"TOTALE: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"{'='*70}")
    return {"precision": prec, "recall": rec, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def run_single(backend_name: str, **overrides):
    print(f"\n{'#'*70}")
    print(f"  TESTING: {backend_name.upper()}")
    print(f"{'#'*70}\n")
    cfg = Config(backend=backend_name, **overrides)
    rec = BookRecognizer(cfg)
    rec.load_library("./dataset/models/")
    all_results = rec.run_with_eval("./dataset/scenes/")
    return evaluate_results(all_results)


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "sift_lg"

    if backend == "all":
        results = {}
        for b in ["sift_lg", "superpoint", "aliked"]:
            try:
                results[b] = run_single(b, output_dir=f"output_hybrid_{b}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[b] = {"precision": 0, "recall": 0, "f1": 0}

        print(f"\n\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"{'Backend':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 50)
        for b, m in results.items():
            print(f"{b:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
        print(f"{'='*70}")
    else:
        run_single(backend, output_dir=f"output_hybrid_{backend}")
