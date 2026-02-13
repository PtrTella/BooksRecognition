"""
Book Spine Recognition -- Star Model (Generalised Hough Transform)
=================================================================

Pipeline
--------
Preprocess -> SIFT/RootSIFT -> FLANN match -> scale-filter -> GHT vote
-> Affine/Homography verify (peeling) -> NMS

Classical GHT-of-SIFT: each model keypoint stores a displacement vector to
the object centre, normalised by scale and derotated by orientation.  At
detection time, matches vote for centre positions; peaks are candidate
instances.

Geometric verification uses a two-tier strategy:
  - With few matches (< affine_thresh): partial-affine (similarity, 4 DOF)
    via ``estimateAffinePartial2D``.  Needs only 2 correspondences, ideal
    for thin book spines that yield sparse features.
  - With many matches (>= affine_thresh): full RANSAC homography.
    More selective for feature-rich models.

A peeling loop extracts multiple instances from one GHT peak.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fclusterdata


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All tuneable parameters in one place."""

    # Preprocessing (CLAHE + unsharp mask on LAB L-channel)
    clahe_clip:         float = 2.5
    clahe_grid:         int   = 8
    sharpen_amount:     float = 0.8     # unsharp-mask strength

    # SIFT
    nfeatures:          int   = 0       # 0 = no cap
    contrast_threshold: float = 0.03
    edge_threshold:     float = 10
    n_octave_layers:    int   = 3
    root_sift:          bool  = True

    # FLANN matching
    lowe_ratio:    float = 0.78
    flann_trees:   int   = 5
    flann_checks:  int   = 100
    scale_sigma:   float = 2.5   # reject matches whose scale ratio deviates
                                 # more than this many MADs from the median

    # GHT voting
    bin_size:     int = 20       # smaller bins => better separation of close instances
    merge_radius: int = 1
    min_votes:    int = 4

    # Geometric verification
    ransac_thresh:   float = 5.0
    min_inliers:     int   = 5
    min_inliers_h:   int   = 8   # stricter for full homography
    affine_thresh:   int   = 12  # use partial-affine when #matches < this
    min_det:         float = 0.02
    max_det:         float = 50.0
    min_area:        int   = 500
    max_area_frac:   float = 0.9

    # NMS
    iou_same:  float = 0.50  # Permette detection più vicine per stesso modello
    iou_cross: float = 0.50


# ---------------------------------------------------------------------------
#  Preprocessing
# ---------------------------------------------------------------------------

class Preprocessor:
    """Enhance images for robust SIFT feature extraction.

    Returns a single-channel (grayscale) image:
      1. CLAHE on the L channel of CIE-LAB -- local contrast
         enhancement that reveals detail in dark / bright regions.
      2. Unsharp mask -- gentle sharpening to boost gradient
         responses for SIFT, especially on thin book spines.
    
    NOTE: Bilateral filter and morphological operations were tested
    but reduced keypoints by 30%, hurting recall.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
        )

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        # CLAHE on L channel
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        l = self._clahe.apply(l)

        # Unsharp mask:  sharpened = L + amount * (L - blur)
        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=2.0)
        alpha = self._cfg.sharpen_amount
        l = cv2.addWeighted(l, 1.0 + alpha, blur, -alpha, 0)
        return l


# ---------------------------------------------------------------------------
#  Feature Index
# ---------------------------------------------------------------------------

class FeatureIndex:
    """SIFT / RootSIFT extraction + global FLANN index over all models.

    All model descriptors are added with ``add``; once ``build`` is called
    a single ``knnMatch`` per scene returns matches grouped by book_id
    through ``imgIdx``.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._sift = cv2.SIFT_create(
            nfeatures=cfg.nfeatures,
            contrastThreshold=cfg.contrast_threshold,
            edgeThreshold=cfg.edge_threshold,
            nOctaveLayers=cfg.n_octave_layers,
        )
        self._ids: List[str] = []
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=cfg.flann_trees),
            dict(checks=cfg.flann_checks),
        )

    def extract(
        self, gray: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect keypoints and compute (Root)SIFT descriptors."""
        kp, des = self._sift.detectAndCompute(gray, None)
        if des is None:
            return [], None
        des = des.astype(np.float32)
        if self.cfg.root_sift:
            des /= des.sum(axis=1, keepdims=True) + 1e-7
            np.sqrt(des, out=des)
        return kp, des

    def add(self, book_id: str, des: np.ndarray) -> None:
        self._matcher.add([des])
        self._ids.append(book_id)

    def build(self) -> None:
        self._matcher.train()

    def match(
        self, des_scene: np.ndarray, kp_scene: List[cv2.KeyPoint],
        model_kps: Dict[str, List[cv2.KeyPoint]],
    ) -> Dict[str, list]:
        """Return Lowe-ratio-filtered, scale-consistent matches grouped by book_id.

        After ratio filtering, matches whose scene/model keypoint size
        ratio deviates by more than ``scale_sigma`` MADs from the group
        median are removed.  This suppresses outlier matches whose scale
        is physically implausible.
        """
        if des_scene is None or not self._ids:
            return {}
        raw = self._matcher.knnMatch(des_scene, k=2)
        groups: Dict[str, list] = defaultdict(list)
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg.lowe_ratio * n.distance:
                groups[self._ids[m.imgIdx]].append(m)

        # Scale-ratio consistency filter per model
        sigma = self.cfg.scale_sigma
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


# ---------------------------------------------------------------------------
#  Star Model  (Generalised Hough Transform + Homography verification)
# ---------------------------------------------------------------------------

class StarModel:
    """Classical GHT-of-SIFT for multi-instance object detection.

    Registration
        Each model keypoint records a displacement vector to the model
        centre, derotated by ``R(-angle)`` and divided by ``scale`` so the
        vector becomes orientation- and scale-invariant.

    Detection
        1. Each match casts a centre vote (re-rotating and re-scaling the
           stored vector).
        2. Votes are binned in a 2-D accumulator; neighbouring bins are
           merged into peaks (candidate instances).
        3. Within each peak, a peeling loop iteratively applies geometric
           verification (partial-affine for sparse peaks, full homography
           for dense ones), strips the inliers, and repeats to extract
           multiple instances from one spatial cluster.
        4. Greedy NMS with two-tier IoU thresholds (stricter for
           same-model duplicates) removes redundant boxes.
    """

    def __init__(self, cfg: Config, index: FeatureIndex) -> None:
        self.cfg = cfg
        self.index = index
        self._tpl: Dict[str, dict] = {}

    # -- registration ----------------------------------------------------------

    def register(
        self, book_id: str, kp: List[cv2.KeyPoint], shape: Tuple[int, ...],
    ) -> None:
        """Store per-keypoint displacement vectors to the model centre."""
        h, w = shape[:2]
        cx, cy = w / 2.0, h / 2.0

        pts = np.array([k.pt for k in kp], dtype=np.float64)
        angles = np.deg2rad([k.angle for k in kp])
        sizes = np.array([k.size for k in kp])

        d = np.array([cx, cy]) - pts                            # (N, 2)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        vecs = np.column_stack([
             cos_a * d[:, 0] + sin_a * d[:, 1],
            -sin_a * d[:, 0] + cos_a * d[:, 1],
        ]) / sizes[:, np.newaxis]                               # (N, 2)

        self._tpl[book_id] = {"kp": kp, "vecs": vecs, "shape": shape}

    # -- detection -------------------------------------------------------------

    def detect(
        self,
        kp_scene: List[cv2.KeyPoint],
        des_scene: Optional[np.ndarray],
        scene_shape: Tuple[int, ...],
    ) -> List[dict]:
        if des_scene is None:
            return []
        # Collect model keypoints for scale-ratio filtering
        model_kps = {bid: tpl["kp"] for bid, tpl in self._tpl.items()}
        cands: List[dict] = []
        for bid, matches in self.index.match(des_scene, kp_scene, model_kps).items():
            tpl = self._tpl[bid]
            if len(matches) < self.cfg.min_votes:
                continue
            for peak in self._vote(matches, kp_scene, tpl, scene_shape):
                cands.extend(
                    self._extract(bid, peak, tpl, kp_scene, scene_shape),
                )
        return self._nms(cands)

    # -- GHT voting ------------------------------------------------------------

    def _vote(
        self,
        matches: list,
        kp_scene: List[cv2.KeyPoint],
        tpl: dict,
        scene_shape: Tuple[int, ...],
    ) -> List[list]:
        """Cast centre votes and cluster spatially for multi-instance detection."""
        vecs = tpl["vecs"]

        qidx = [m.queryIdx for m in matches]
        tidx = [m.trainIdx for m in matches]
        pts = np.array([kp_scene[i].pt for i in qidx])
        angles = np.deg2rad([kp_scene[i].angle for i in qidx])
        sizes = np.array([kp_scene[i].size for i in qidx])
        stored = vecs[tidx]

        cos_a, sin_a = np.cos(angles), np.sin(angles)
        votes = pts + np.column_stack([
            cos_a * stored[:, 0] - sin_a * stored[:, 1],
            sin_a * stored[:, 0] + cos_a * stored[:, 1],
        ]) * sizes[:, np.newaxis]

        valid = ((votes[:, 0] >= 0) & (votes[:, 0] < scene_shape[1]) &
                 (votes[:, 1] >= 0) & (votes[:, 1] < scene_shape[0]))
        
        valid_votes = votes[valid]
        valid_matches = [matches[i] for i in np.flatnonzero(valid)]
        
        if len(valid_votes) < self.cfg.min_votes:
            return []
        
        # Use hierarchical clustering instead of rigid binning
        # This better separates adjacent identical books
        cluster_dist = self.cfg.bin_size * 2
        clusters = fclusterdata(valid_votes, t=cluster_dist, criterion='distance', method='single')
        
        # Group matches by cluster
        groups: Dict[int, list] = defaultdict(list)
        for i, c in enumerate(clusters):
            groups[c].append(valid_matches[i])
        
        # Filter by min_votes and sort by size
        peaks = [g for g in groups.values() if len(g) >= self.cfg.min_votes]
        peaks.sort(key=len, reverse=True)
        
        return peaks

    # -- instance extraction (peeling) -----------------------------------------

    def _extract(
        self,
        bid: str,
        matches: list,
        tpl: dict,
        kp_scene: list,
        scene_shape: Tuple[int, ...],
    ) -> List[dict]:
        """Peel one or more instances out of a GHT peak.

        Iteratively fit geometric model, collect the result, strip
        inliers from the pool, repeat.  Uses partial-affine for sparse
        peaks (< affine_thresh matches) and full homography otherwise.
        """
        found: List[dict] = []
        pool = list(matches)
        while len(pool) >= self.cfg.min_inliers:
            r = self._verify(bid, pool, tpl, kp_scene, scene_shape)
            if r is None:
                break
            found.append(r)
            pool = self._strip(pool, tpl, kp_scene, r["H"])
        return found

    def _strip(
        self,
        matches: list,
        tpl: dict,
        kp_scene: list,
        H: np.ndarray,
    ) -> list:
        """Remove inliers of *H* from the match pool.

        Works for both 3×3 homography and 2×3 affine matrices.
        """
        src = np.float32(
            [tpl["kp"][m.trainIdx].pt for m in matches],
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [kp_scene[m.queryIdx].pt for m in matches],
        ).reshape(-1, 2)
        if H.shape[0] == 2:
            # 2×3 affine
            proj = cv2.transform(src, H).reshape(-1, 2)
        else:
            proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        return [m for m, e in zip(matches, err)
                if e >= self.cfg.ransac_thresh * 1.5]

    # -- geometric verification ------------------------------------------------

    def _verify(
        self,
        bid: str,
        matches: list,
        tpl: dict,
        kp_scene: list,
        scene_shape: Tuple[int, ...],
    ) -> Optional[dict]:
        """Two-tier geometric verification at a GHT peak.

        - Sparse peaks (< affine_thresh matches): ``estimateAffinePartial2D``
          (similarity transform, 4 DOF).  Needs only 2 correspondences,
          making it far more reliable than homography for thin book spines
          with few features.
        - Dense peaks (>= affine_thresh matches): full RANSAC homography.

        Both paths validate: sufficient inliers, sane determinant, convex
        projected quadrilateral, reasonable area.
        """
        src = np.float32(
            [tpl["kp"][m.trainIdx].pt for m in matches],
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [kp_scene[m.queryIdx].pt for m in matches],
        ).reshape(-1, 1, 2)

        use_affine = len(matches) < self.cfg.affine_thresh

        if use_affine:
            # Similarity transform (scale + rotation + translation)
            M, mask = cv2.estimateAffinePartial2D(
                src, dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.ransac_thresh,
            )
            if M is None or mask is None:
                return None
            H = M  # 2×3
        else:
            H, mask = cv2.findHomography(
                src, dst, cv2.RANSAC, self.cfg.ransac_thresh,
            )
            if H is None or mask is None:
                return None

        n_inliers = int(mask.ravel().sum())
        # Stricter threshold for full homography (8 DOF more prone to
        # overfitting with few points) vs partial-affine (4 DOF, robust)
        min_inl = self.cfg.min_inliers if use_affine else self.cfg.min_inliers_h
        if n_inliers < min_inl:
            return None
        
        # Adaptive score threshold based on model complexity
        model_kp_count = len(tpl["kp"])
        score_threshold = max(3, int(model_kp_count * 0.005))  # 0.5% of model keypoints
        if n_inliers < score_threshold:
            return None

        # Determinant sanity (reject flips / extreme warps)
        det = np.linalg.det(H[:2, :2])
        if det < self.cfg.min_det or det > self.cfg.max_det:
            return None

        # Project model corners
        h, w = tpl["shape"][:2]
        corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]],
        ).reshape(-1, 1, 2)
        if use_affine:
            bbox = cv2.transform(corners, H)
        else:
            bbox = cv2.perspectiveTransform(corners, H)

        # Convexity
        if not cv2.isContourConvex(bbox.astype(np.int32)):
            return None

        # Aspect-ratio consistency: the detected quad's aspect ratio
        # should not deviate wildly from the model's.
        pts4 = bbox.reshape(4, 2)
        w_det = (np.linalg.norm(pts4[1] - pts4[0])
                 + np.linalg.norm(pts4[2] - pts4[3])) / 2.0
        h_det = (np.linalg.norm(pts4[3] - pts4[0])
                 + np.linalg.norm(pts4[2] - pts4[1])) / 2.0
        if w_det < 1 or h_det < 1:
            return None
        ar_det = max(w_det, h_det) / min(w_det, h_det)
        ar_mod = max(w, h) / max(min(w, h), 1)
        if ar_det < ar_mod * 0.3 or ar_det > ar_mod * 3.0:
            return None

        # Area
        area = cv2.contourArea(bbox)
        scene_area = scene_shape[0] * scene_shape[1]
        if area < self.cfg.min_area or area > scene_area * self.cfg.max_area_frac:
            return None

        return {
            "book_id": bid,
            "score": n_inliers,
            "bbox": bbox,
            "H": H,       # original 2×3 or 3×3 for stripping
        }

    # -- NMS -------------------------------------------------------------------

    def _nms(self, cands: List[dict]) -> List[dict]:
        """Score-ordered greedy NMS with two-tier IoU thresholds."""
        if not cands:
            return []
        cands.sort(key=lambda c: c["score"], reverse=True)
        kept: List[dict] = []
        for c in cands:
            cp = c["bbox"].reshape(4, 2).astype(np.float32)
            dup = False
            for k in kept:
                kp_ = k["bbox"].reshape(4, 2).astype(np.float32)
                thresh = (self.cfg.iou_same if c["book_id"] == k["book_id"]
                          else self.cfg.iou_cross)
                if self._iou(cp, kp_) > thresh:
                    dup = True
                    break
            if not dup:
                kept.append(c)
        return kept

    @staticmethod
    def _iou(p1: np.ndarray, p2: np.ndarray) -> float:
        ok, inter = cv2.intersectConvexConvex(p1, p2)
        if not ok or inter is None:
            return 0.0
        ia = cv2.contourArea(inter)
        return ia / (cv2.contourArea(p1) + cv2.contourArea(p2) - ia + 1e-7)


# ---------------------------------------------------------------------------
#  Pipeline Orchestrator
# ---------------------------------------------------------------------------

class BookRecognizer:
    """Load models, process scenes, visualise results."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.prep = Preprocessor(self.cfg)
        self.index = FeatureIndex(self.cfg)
        self.star = StarModel(self.cfg, self.index)

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
            kp, des = self.index.extract(enhanced)
            if des is None:
                continue
            self.star.register(bid, kp, enhanced.shape)
            self.index.add(bid, des)
            n += 1
            print(f"  {bid}: {len(kp)} keypoints")
        self.index.build()
        print(f"{n} models indexed.\n")

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
            kp, des = self.index.extract(enhanced)
            results = self.star.detect(kp, des, enhanced.shape)
            self._draw(img, results, f)
            total += len(results)
        print(f"\nTotal: {total} books detected across {len(files)} scenes")

    @staticmethod
    def _draw(img: np.ndarray, results: List[dict], name: str) -> None:
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
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA,
            )

        # Save to file AND show
        import os
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", f"result_{name}")
        cv2.imwrite(out_path, vis)
        
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"{name} -- {len(results)} book(s): {ids}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        for r in results:
            print(f"  -> {r['book_id']} ({r['score']} inliers)")


# ---------------------------------------------------------------------------
#  Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rec = BookRecognizer()
    rec.load_library("./dataset/models/")
    rec.run("./dataset/scenes/")
