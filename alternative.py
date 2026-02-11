"""
Book Spine Recognition -- Hybrid Pipeline
==========================================

SIFT/RootSIFT → FLANN → Star-Model centre votes → DBSCAN spatial
clustering → Affine peeling per cluster → NMS

Classes are organised by single responsibility:

  Config             -- all tuneable parameters
  Preprocessor       -- CLAHE + unsharp mask → grayscale
  FeatureExtractor   -- SIFT / RootSIFT keypoint & descriptor computation
  FeatureMatcher     -- FLANN index, Lowe ratio test, MAD scale filter
  StarVoter          -- star-model registration & centre-vote projection
  Clusterer          -- DBSCAN spatial grouping of projected centres
  GeometricVerifier  -- affine peeling: fit, validate, strip inliers
  Visualizer         -- draw bounding boxes on scene images
  BookRecognizer     -- thin orchestrator wiring everything together
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """All tuneable parameters in one place."""

    # -- Preprocessing (CLAHE + unsharp mask on LAB L-channel) --
    clahe_clip:         float = 2.5
    clahe_grid:         int   = 8
    sharpen_amount:     float = 0.8

    # -- SIFT --
    nfeatures:          int   = 0
    contrast_threshold: float = 0.03
    edge_threshold:     float = 10
    n_octave_layers:    int   = 3
    root_sift:          bool  = True

    # -- FLANN matching --
    lowe_ratio:    float = 0.78
    flann_trees:   int   = 5
    flann_checks:  int   = 100
    scale_sigma:   float = 2.5     # MAD outlier rejection

    # -- DBSCAN clustering --
    dbscan_eps:      float = 20.0  # neighbourhood radius (px)
    dbscan_min_pts:  int   = 3     # core-point threshold

    # -- Geometric verification --
    ransac_thresh:   float = 5.0
    min_inliers:     int   = 5
    min_det:         float = 0.02  # determinant lower bound
    max_det:         float = 50.0  # determinant upper bound
    min_area:        int   = 500
    max_area_frac:   float = 0.9
    ar_tolerance:    float = 3.0   # aspect-ratio tolerance factor

    # -- NMS --
    iou_same:  float = 0.40
    iou_cross: float = 0.50


# ═══════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """Enhance images for robust feature extraction.

    1. CLAHE on the L channel of CIE-LAB for local contrast.
    2. Unsharp mask for gentle sharpening.
    Returns single-channel grayscale.
    """

    def __init__(self, cfg: Config) -> None:
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
#  Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """SIFT / RootSIFT keypoint detection and descriptor computation."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._sift = cv2.SIFT_create(
            nfeatures=cfg.nfeatures,
            contrastThreshold=cfg.contrast_threshold,
            edgeThreshold=cfg.edge_threshold,
            nOctaveLayers=cfg.n_octave_layers,
        )

    def __call__(
        self, gray: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        kp, des = self._sift.detectAndCompute(gray, None)
        if des is None:
            return [], None
        des = des.astype(np.float32)
        if self._cfg.root_sift:
            des /= des.sum(axis=1, keepdims=True) + 1e-7
            np.sqrt(des, out=des)
        return kp, des


# ═══════════════════════════════════════════════════════════════════════════
#  Feature Matching
# ═══════════════════════════════════════════════════════════════════════════

class FeatureMatcher:
    """Global FLANN index over all models.

    Matches are filtered by Lowe's ratio test and MAD-based
    scale-ratio consistency.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._ids: List[str] = []
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=cfg.flann_trees),
            dict(checks=cfg.flann_checks),
        )

    def add(self, book_id: str, des: np.ndarray) -> None:
        """Add a model's descriptors to the index."""
        self._matcher.add([des])
        self._ids.append(book_id)

    def build(self) -> None:
        """Train the FLANN index (call after all models are added)."""
        self._matcher.train()

    def match(
        self,
        des_scene: Optional[np.ndarray],
        kp_scene: List[cv2.KeyPoint],
        model_kps: Dict[str, List[cv2.KeyPoint]],
    ) -> Dict[str, list]:
        """Return Lowe-filtered, scale-consistent matches by book_id."""
        if des_scene is None or not self._ids:
            return {}

        raw = self._matcher.knnMatch(des_scene, k=2)

        # Lowe ratio test
        groups: Dict[str, list] = defaultdict(list)
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self._cfg.lowe_ratio * n.distance:
                groups[self._ids[m.imgIdx]].append(m)

        # MAD-based scale-ratio filter per model
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
            ratios = np.array([
                kp_scene[m.queryIdx].size / mkp[m.trainIdx].size
                for m in ms
            ])
            med = np.median(ratios)
            mad = np.median(np.abs(ratios - med)) + 1e-7
            keep = np.abs(ratios - med) <= sigma * mad
            filtered[bid] = [m for m, k in zip(ms, keep) if k]

        return filtered


# ═══════════════════════════════════════════════════════════════════════════
#  Star-Model Voter  (centre-vote projection)
# ═══════════════════════════════════════════════════════════════════════════

class StarVoter:
    """Star-model registration and centre-vote projection.

    Each model keypoint stores a displacement vector to the model
    centre, derotated by the keypoint orientation and normalised by
    its scale.  At detection time, matched scene keypoints re-rotate
    and re-scale the stored vector to cast a centre vote.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, dict] = {}

    @property
    def templates(self) -> Dict[str, dict]:
        return self._templates

    def register(
        self,
        book_id: str,
        kp: List[cv2.KeyPoint],
        shape: Tuple[int, ...],
    ) -> None:
        """Compute and store per-keypoint displacement vectors."""
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

        self._templates[book_id] = {
            "kp": kp, "vecs": vecs, "shape": shape,
        }

    def project(
        self,
        matches: list,
        kp_scene: List[cv2.KeyPoint],
        tpl: dict,
    ) -> np.ndarray:
        """Project each match to a 2-D centre vote in scene space."""
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


# ═══════════════════════════════════════════════════════════════════════════
#  DBSCAN Clusterer
# ═══════════════════════════════════════════════════════════════════════════

class Clusterer:
    """Density-based spatial clustering of 2-D centre votes.

    Returns a dict mapping cluster label → list of match indices.
    Noise points (label == -1) are discarded.
    """

    def __init__(self, cfg: Config) -> None:
        self._eps = cfg.dbscan_eps
        self._min_pts = cfg.dbscan_min_pts

    def __call__(self, centres: np.ndarray) -> Dict[int, List[int]]:
        db = DBSCAN(
            eps=self._eps,
            min_samples=self._min_pts,
        ).fit(centres)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(db.labels_):
            if label >= 0:
                clusters[label].append(i)
        return clusters


# ═══════════════════════════════════════════════════════════════════════════
#  Geometric Verifier  (affine peeling)
# ═══════════════════════════════════════════════════════════════════════════

class GeometricVerifier:
    """Partial-affine (similarity, 4 DOF) fitting with peeling.

    Given a pool of matches from one DBSCAN cluster, iteratively:
      1. Fit a similarity transform via ``estimateAffinePartial2D``.
      2. Validate: inlier count, determinant, convexity, aspect ratio,
         area.
      3. Record the detection and strip its inliers from the pool.
      4. Repeat until the pool is exhausted.
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
        """Extract one or more instances from a match cluster."""
        found: List[dict] = []
        pool = list(matches)
        while len(pool) >= self._cfg.min_inliers:
            result = self._fit_and_validate(
                bid, pool, tpl, kp_scene, scene_shape,
            )
            if result is None:
                break
            found.append(result)
            pool = self._strip_inliers(pool, tpl, kp_scene, result["H"])
        return found

    def _fit_and_validate(
        self,
        bid: str,
        matches: list,
        tpl: dict,
        kp_scene: List[cv2.KeyPoint],
        scene_shape: Tuple[int, ...],
    ) -> Optional[dict]:
        """Fit partial-affine and run all sanity checks."""
        src = np.float32(
            [tpl["kp"][m.trainIdx].pt for m in matches],
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [kp_scene[m.queryIdx].pt for m in matches],
        ).reshape(-1, 1, 2)

        H, mask = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._cfg.ransac_thresh,
        )
        if H is None or mask is None:
            return None

        n_inliers = int(mask.ravel().sum())
        if n_inliers < self._cfg.min_inliers:
            return None

        # -- determinant (reject flips / extreme warps) --
        det = np.linalg.det(H[:2, :2])
        if det < self._cfg.min_det or det > self._cfg.max_det:
            return None

        # -- project model corners → bounding quad --
        h, w = tpl["shape"][:2]
        corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]],
        ).reshape(-1, 1, 2)
        bbox = cv2.transform(corners, H)

        if not cv2.isContourConvex(bbox.astype(np.int32)):
            return None

        # -- aspect-ratio consistency --
        pts4 = bbox.reshape(4, 2)
        w_det = (np.linalg.norm(pts4[1] - pts4[0])
                 + np.linalg.norm(pts4[2] - pts4[3])) / 2.0
        h_det = (np.linalg.norm(pts4[3] - pts4[0])
                 + np.linalg.norm(pts4[2] - pts4[1])) / 2.0
        if w_det < 1 or h_det < 1:
            return None
        ar_det = max(w_det, h_det) / min(w_det, h_det)
        ar_mod = max(w, h) / max(min(w, h), 1)
        tol = self._cfg.ar_tolerance
        if ar_det < ar_mod / tol or ar_det > ar_mod * tol:
            return None

        # -- area bounds --
        area = cv2.contourArea(bbox)
        scene_area = scene_shape[0] * scene_shape[1]
        if area < self._cfg.min_area or area > scene_area * self._cfg.max_area_frac:
            return None

        return {"book_id": bid, "score": n_inliers, "bbox": bbox, "H": H}

    def _strip_inliers(self, matches, tpl, kp_scene, H):
        """Remove matches consistent with affine *H* (within 1.5× threshold)."""
        src = np.float32(
            [tpl["kp"][m.trainIdx].pt for m in matches],
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [kp_scene[m.queryIdx].pt for m in matches],
        ).reshape(-1, 2)
        proj = cv2.transform(src, H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        return [m for m, e in zip(matches, err)
                if e >= self._cfg.ransac_thresh * 1.5]


# ═══════════════════════════════════════════════════════════════════════════
#  Non-Maximum Suppression
# ═══════════════════════════════════════════════════════════════════════════

def nms(cands: List[dict], cfg: Config) -> List[dict]:
    """Score-ordered greedy NMS with two-tier IoU thresholds
    (stricter for same-model duplicates).
    """
    if not cands:
        return []
    cands.sort(key=lambda c: c["score"], reverse=True)
    kept: List[dict] = []
    for c in cands:
        cp = c["bbox"].reshape(4, 2).astype(np.float32)
        dup = False
        for k in kept:
            kp_ = k["bbox"].reshape(4, 2).astype(np.float32)
            thresh = (cfg.iou_same if c["book_id"] == k["book_id"]
                      else cfg.iou_cross)
            if _iou(cp, kp_) > thresh:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def _iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """Intersection-over-union for two convex polygons."""
    ok, inter = cv2.intersectConvexConvex(p1, p2)
    if not ok or inter is None:
        return 0.0
    ia = cv2.contourArea(inter)
    return ia / (cv2.contourArea(p1) + cv2.contourArea(p2) - ia + 1e-7)


# ═══════════════════════════════════════════════════════════════════════════
#  Visualizer
# ═══════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Draw detection bounding boxes on scene images."""

    @staticmethod
    def draw(img: np.ndarray, results: List[dict], name: str) -> None:
        vis = img.copy()
        rng = np.random.RandomState(42)
        n = max(len(results), 1)
        colors = [
            tuple(int(c) for c in row)
            for row in rng.randint(50, 230, (n, 3)).tolist()
        ]
        ids: List[str] = []
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
            (tw, thr), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, fs, th,
            )
            pts = bbox.reshape(-1, 2)
            x, y = pts[pts[:, 1].argmin()]
            x = max(0, min(x, vis.shape[1] - tw))
            y = max(thr + 12, y)
            cv2.rectangle(
                vis, (x, y - thr - 8), (x + tw + 4, y + 2), col, -1,
            )
            cv2.putText(
                vis, label, (x + 2, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th,
                cv2.LINE_AA,
            )

        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(
            f"{name} -- {len(results)} book(s): {', '.join(ids)}",
            fontsize=10,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        for r in results:
            print(f"  -> {r['book_id']} ({r['score']} inliers)")


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class BookRecognizer:
    """Thin orchestrator wiring all components together.

    Load models → for each scene: preprocess, extract features,
    match, project centres, cluster, verify/peel, NMS, visualise.
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.prep      = Preprocessor(self.cfg)
        self.extractor = FeatureExtractor(self.cfg)
        self.matcher   = FeatureMatcher(self.cfg)
        self.voter     = StarVoter()
        self.clusterer = Clusterer(self.cfg)
        self.verifier  = GeometricVerifier(self.cfg)
        self.viz       = Visualizer()

    def load_library(self, path: str) -> None:
        """Index all model images in *path*."""
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
            kp, des = self.extractor(enhanced)
            if des is None:
                continue
            self.voter.register(bid, kp, enhanced.shape)
            self.matcher.add(bid, des)
            n += 1
            print(f"  {bid}: {len(kp)} keypoints")
        self.matcher.build()
        print(f"{n} models indexed.\n")

    def detect(
        self,
        kp_scene: List[cv2.KeyPoint],
        des_scene: Optional[np.ndarray],
        scene_shape: Tuple[int, ...],
    ) -> List[dict]:
        """Run the full detection pipeline on one scene."""
        if des_scene is None:
            return []

        model_kps = {
            bid: tpl["kp"]
            for bid, tpl in self.voter.templates.items()
        }
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
                cands.extend(
                    self.verifier.peel(
                        bid, cluster_matches, tpl, kp_scene, scene_shape,
                    ),
                )

        return nms(cands, self.cfg)

    def run(self, scenes_path: str) -> None:
        """Run detection on every scene image in *scenes_path*."""
        files = sorted(
            f for f in os.listdir(scenes_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        print(f"Processing {len(files)} scenes ...")
        total = 0
        for f in files:
            img = cv2.imread(os.path.join(scenes_path, f))
            if img is None:
                continue
            enhanced = self.prep(img)
            kp, des = self.extractor(enhanced)
            results = self.detect(kp, des, enhanced.shape)
            self.viz.draw(img, results, f)
            total += len(results)
        print(f"\nTotal: {total} books detected across {len(files)} scenes")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rec = BookRecognizer()
    rec.load_library("./dataset/models/")
    rec.run("./dataset/scenes/")
