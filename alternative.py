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
    clahe_clip:         float = 3.0      # Più aggressivo per testo su spine
    clahe_grid:         int   = 8
    sharpen_amount:     float = 1.0      # Più sharpening

    # -- SIFT -- più sensibile per più keypoints
    nfeatures:          int   = 0
    contrast_threshold: float = 0.01     # Più basso = più keypoints
    edge_threshold:     float = 15       # Più alto = meno edge rejection
    n_octave_layers:    int   = 4        # Più layers = più scale
    root_sift:          bool  = True

    # -- FLANN matching --
    lowe_ratio:    float = 0.78          # Stretto per meno FP
    flann_trees:   int   = 5
    flann_checks:  int   = 100
    scale_sigma:   float = 3.5           # Più tollerante su scale differences

    # -- DBSCAN clustering --
    dbscan_eps:      float = 20.0  # neighbourhood radius (px)
    dbscan_min_pts:  int   = 3     # core-point threshold

    # -- Geometric verification --
    ransac_thresh:   float = 4.0         # Leggermente più stretto
    min_inliers:     int   = 3           # Abbassato per spine sottili
    min_det:         float = 0.025       # determinant lower bound
    max_det:         float = 40.0        # determinant upper bound
    min_area:        int   = 450
    max_area_frac:   float = 0.9
    ar_tolerance:    float = 3.0   # aspect-ratio tolerance factor

    # -- NMS --
    iou_same:  float = 0.30
    iou_cross: float = 0.35

    # -- Edge density verification (post-filter) --
    edge_verify:    bool  = True   # Enable edge density verification
    edge_tolerance: float = 2.6    # Tolerance factor for edge density check

    # -- NCC warp-and-compare verification --
    ncc_verify:     bool  = True
    ncc_threshold:  float = 0.20   # Reject detections with NCC below this


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
            
            # Safely compute scale ratios (handles rotation augmentation)
            valid_ms = []
            ratios = []
            for m in ms:
                if m.trainIdx < len(mkp):
                    valid_ms.append(m)
                    ratios.append(kp_scene[m.queryIdx].size / mkp[m.trainIdx].size)
            
            if len(valid_ms) < 3:
                filtered[bid] = ms  # Not enough for MAD, keep all
                continue
            
            ratios = np.array(ratios)
            med = np.median(ratios)
            mad = np.median(np.abs(ratios - med)) + 1e-7
            keep = np.abs(ratios - med) <= sigma * mad
            filtered[bid] = [m for m, k in zip(valid_ms, keep) if k]

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
    """DBSCAN spatial clustering of 2-D centre votes.

    Returns a dict mapping cluster label → list of match indices.
    Noise points (label == -1) are discarded.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def __call__(self, centres: np.ndarray) -> Dict[int, List[int]]:
        if len(centres) < 2:
            return {0: list(range(len(centres)))} if len(centres) else {}

        db = DBSCAN(
            eps=self._cfg.dbscan_eps,
            min_samples=self._cfg.dbscan_min_pts,
        )
        labels = db.fit_predict(centres)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
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
#  Edge Density Verification
# ═══════════════════════════════════════════════════════════════════════════

class EdgeDensityVerifier:
    """Verify detections by comparing edge density between model and scene ROI.
    
    Rejects detections where the edge density ratio differs too much,
    which helps filter false positives from texture-poor regions.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._templates: Dict[str, float] = {}

    def register(self, bid: str, enhanced: np.ndarray) -> None:
        """Store edge density for a model image."""
        edges = cv2.Canny(enhanced, 50, 150)
        self._templates[bid] = np.mean(edges > 0)

    def verify(self, det: dict, scene_enhanced: np.ndarray) -> bool:
        """Return True if detection passes edge density check."""
        bid = det["book_id"]
        if bid not in self._templates:
            return True  # No template, skip verification
        
        # Extract scene ROI and compute edge density
        mask = np.zeros(scene_enhanced.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, det["bbox"].astype(np.int32), 255)
        scene_edges = cv2.Canny(scene_enhanced, 50, 150)
        
        roi_pixels = np.sum(mask > 0)
        if roi_pixels == 0:
            return True
        
        scene_density = np.sum((scene_edges > 0) & (mask > 0)) / roi_pixels
        model_density = self._templates[bid]
        
        if model_density < 1e-6:
            return True  # Avoid division by zero
        
        ratio = scene_density / model_density
        tol = self._cfg.edge_tolerance
        return (1.0 / tol) <= ratio <= tol


# ═══════════════════════════════════════════════════════════════════════════
#  NCC Warp-and-Compare Verification
# ═══════════════════════════════════════════════════════════════════════════

class NCCVerifier:
    """Warp model into scene space and compute Normalised Cross Correlation.

    Catches false positives: when SIFT finds a consistent affine transform
    but the warped model's pixel content doesn't actually match the scene
    region, the detection is likely spurious.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._model_images: Dict[str, np.ndarray] = {}

    def register(self, bid: str, enhanced: np.ndarray) -> None:
        self._model_images[bid] = enhanced

    def verify(self, det: dict, scene_enhanced: np.ndarray) -> bool:
        bid = det["book_id"]
        model_img = self._model_images.get(bid)
        if model_img is None:
            return True
        H = det["H"]
        h_s, w_s = scene_enhanced.shape[:2]

        if H.shape[0] == 2:
            warped = cv2.warpAffine(model_img, H, (w_s, h_s),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpAffine(np.ones_like(model_img) * 255, H, (w_s, h_s),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            warped = cv2.warpPerspective(model_img, H, (w_s, h_s),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpPerspective(np.ones_like(model_img) * 255, H, (w_s, h_s),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        valid = mask > 128
        if valid.sum() < 100:
            return True

        w_pix = warped[valid].astype(np.float64)
        s_pix = scene_enhanced[valid].astype(np.float64)
        w_pix -= w_pix.mean()
        s_pix -= s_pix.mean()
        w_std = w_pix.std()
        s_std = s_pix.std()
        if w_std < 1e-8 or s_std < 1e-8:
            return False
        ncc = np.dot(w_pix, s_pix) / (w_std * s_std * len(w_pix))
        return ncc >= self._cfg.ncc_threshold


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
        self.prep         = Preprocessor(self.cfg)
        self.extractor    = FeatureExtractor(self.cfg)
        self.matcher      = FeatureMatcher(self.cfg)
        self.voter        = StarVoter()
        self.clusterer    = Clusterer(self.cfg)
        self.verifier     = GeometricVerifier(self.cfg)
        self.edge_verify  = EdgeDensityVerifier(self.cfg) if self.cfg.edge_verify else None
        self.ncc_verify   = NCCVerifier(self.cfg) if self.cfg.ncc_verify else None
        self.viz          = Visualizer()

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
            if self.edge_verify:
                self.edge_verify.register(bid, enhanced)
            if self.ncc_verify:
                self.ncc_verify.register(bid, enhanced)
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
            
            # Edge density post-verification filter
            if self.edge_verify:
                results = [r for r in results if self.edge_verify.verify(r, enhanced)]

            # NCC warp-and-compare verification
            if self.ncc_verify:
                results = [r for r in results if self.ncc_verify.verify(r, enhanced)]

            self.viz.draw(img, results, f)
            total += len(results)
        print(f"\nTotal: {total} books detected across {len(files)} scenes")


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
    print("EVALUATION vs GROUND TRUTH")
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
    print(f"TOTAL: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"{'='*70}")
    return {"precision": prec, "recall": rec, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rec = BookRecognizer()
    rec.load_library("./dataset/models/")

    # Run with evaluation
    files = sorted(
        f for f in os.listdir("./dataset/scenes/")
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"Processing {len(files)} scenes ...")
    all_results: Dict[str, List[dict]] = {}
    total = 0
    for f in files:
        img = cv2.imread(os.path.join("./dataset/scenes/", f))
        if img is None:
            continue
        enhanced = rec.prep(img)
        kp, des = rec.extractor(enhanced)
        results = rec.detect(kp, des, enhanced.shape)

        # Post-verification filters
        if rec.edge_verify:
            results = [r for r in results if rec.edge_verify.verify(r, enhanced)]
        if rec.ncc_verify:
            results = [r for r in results if rec.ncc_verify.verify(r, enhanced)]

        all_results[f] = results
        total += len(results)

    print(f"\nTotal: {total} books detected across {len(files)} scenes")
    evaluate_results(all_results)
