"""
Book Spine Recognition -- DEBUG VERSION
========================================

Versione con diagnostica completa per capire ogni step della pipeline.
Ottimizzata per libri sottili, vicini e identici.

Problemi comuni:
1. GHT bin_size troppo grande → voti di libri vicini finiscono nello stesso bin
2. merge_radius unisce bin adiacenti → peggiora la separazione
3. Pochi keypoints su spine sottili → difficile raggiungere min_votes
4. Libri identici → stessi match per tutte le istanze, difficile separarli
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import fclusterdata


# ===========================================================================
#  CONFIGURAZIONE OTTIMIZZATA PER LIBRI SOTTILI
# ===========================================================================

@dataclass
class Config:
    """Parametri ottimizzati per spine sottili e multiple istanze."""

    # Preprocessing
    clahe_clip:         float = 3.0      # Più aggressivo per testo su spine
    clahe_grid:         int   = 8
    sharpen_amount:     float = 1.0      # Più sharpening

    # SIFT - più sensibile per catturare più keypoints
    nfeatures:          int   = 0
    contrast_threshold: float = 0.02     # Più basso = più keypoints
    edge_threshold:     float = 15       # Più alto = meno edge rejection
    n_octave_layers:    int   = 4        # Più layers = più scale
    root_sift:          bool  = True

    # Matching - più permissivo
    lowe_ratio:    float = 0.80          # Leggermente più permissivo
    flann_trees:   int   = 5
    flann_checks:  int   = 100
    scale_sigma:   float = 3.0           # Più tollerante su scale differences

    # GHT - CRUCIALE per libri vicini
    bin_size:     int = 12               # RIDOTTO! Per separare libri a ~15px
    merge_radius: int = 0                # ZERO! Non unire bin adiacenti
    min_votes:    int = 3                # Abbassato per spine con pochi keypoints

    # Geometric verification - più permissivo per pochi punti
    ransac_thresh:   float = 4.0         # Leggermente più stretto
    min_inliers:     int   = 4           # Abbassato
    min_inliers_h:   int   = 6           # Abbassato
    affine_thresh:   int   = 10          # Usa affine più spesso
    min_det:         float = 0.01
    max_det:         float = 100.0
    min_area:        int   = 300         # Spine sottili hanno area piccola
    max_area_frac:   float = 0.9

    # NMS - più aggressivo per evitare duplicati
    iou_same:  float = 0.30              # Più stretto per stesso modello
    iou_cross: float = 0.45

    # Debug
    debug: bool = True


# ===========================================================================
#  DIAGNOSTICS - Visualizza ogni step
# ===========================================================================

class Diagnostics:
    """Classe per salvare visualizzazioni su file (non bloccanti)."""
    
    def __init__(self, enabled: bool = True, output_dir: str = "debug_output"):
        self.enabled = enabled
        self.data: Dict[str, Any] = {}
        self.output_dir = output_dir
        self.counter = 0
        if enabled:
            os.makedirs(output_dir, exist_ok=True)
    
    def _save_fig(self, fig, name: str):
        """Salva figura su file invece di mostrarla."""
        self.counter += 1
        path = os.path.join(self.output_dir, f"{self.counter:03d}_{name}.png")
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"      [DEBUG] Salvato: {path}")
        
    def log(self, key: str, value: Any):
        """Salva dati per visualizzazione."""
        if self.enabled:
            self.data[key] = value
    
    def show_preprocessing(self, original: np.ndarray, enhanced: np.ndarray, title: str = ""):
        """Mostra before/after preprocessing."""
        if not self.enabled:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if len(original.shape) == 3:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title(f"Originale {title}")
        axes[0].axis('off')
        
        axes[1].imshow(enhanced, cmap='gray')
        axes[1].set_title(f"Enhanced (CLAHE + Sharpen) {title}")
        axes[1].axis('off')
        
        plt.tight_layout()
        self._save_fig(fig, f"preproc_{title.replace(' ', '_')}")
    
    def show_keypoints(self, img: np.ndarray, kp: List[cv2.KeyPoint], title: str = ""):
        """Mostra keypoints estratti."""
        if not self.enabled:
            return
            
        vis = cv2.drawKeypoints(img, kp, None, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(vis if len(vis.shape) == 3 else vis, cmap='gray' if len(vis.shape) == 2 else None)
        ax.set_title(f"{title} - {len(kp)} keypoints")
        ax.axis('off')
        
        plt.tight_layout()
        self._save_fig(fig, f"keypoints_{title.replace(' ', '_')}")
    
    def show_matches(self, img_model: np.ndarray, kp_model: List[cv2.KeyPoint],
                     img_scene: np.ndarray, kp_scene: List[cv2.KeyPoint],
                     matches: list, title: str = ""):
        """Mostra matches tra modello e scena."""
        if not self.enabled or not matches:
            return
        
        # Converti DMatch in formato corretto
        vis = cv2.drawMatches(img_model, kp_model, img_scene, kp_scene,
                              matches[:50], None,  # Max 50 per visibilità
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(vis)
        ax.set_title(f"{title} - {len(matches)} matches (mostrati max 50)")
        ax.axis('off')
        
        plt.tight_layout()
        self._save_fig(fig, f"matches_{title.replace(' ', '_')}")
    
    def show_votes_heatmap(self, votes: np.ndarray, scene_shape: Tuple[int, int],
                           bin_size: int, peaks: List[Tuple[int, int]], title: str = ""):
        """Mostra heatmap dei voti GHT con peaks identificati."""
        if not self.enabled:
            return
        
        h, w = scene_shape[:2]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter dei voti
        ax1 = axes[0]
        ax1.scatter(votes[:, 0], votes[:, 1], c='red', s=10, alpha=0.5)
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)  # Inverti Y
        ax1.set_title(f"Voti GHT (scatter) - {len(votes)} voti\n{title}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_aspect('equal')
        
        # Griglia con bin
        for x in range(0, w, bin_size):
            ax1.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
        for y in range(0, h, bin_size):
            ax1.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
        
        # Heatmap 2D
        ax2 = axes[1]
        bins_x = int(np.ceil(w / bin_size))
        bins_y = int(np.ceil(h / bin_size))
        heatmap, xedges, yedges = np.histogram2d(
            votes[:, 0], votes[:, 1],
            bins=[bins_x, bins_y],
            range=[[0, w], [0, h]]
        )
        
        im = ax2.imshow(heatmap.T, origin='lower', extent=[0, w, 0, h],
                        cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax2, label='Voti per bin')
        
        # Marca i peaks
        for bx, by in peaks:
            cx = (bx + 0.5) * bin_size
            cy = (by + 0.5) * bin_size
            circle = plt.Circle((cx, cy), bin_size/2, fill=False, 
                                color='cyan', linewidth=2)
            ax2.add_patch(circle)
        
        ax2.set_title(f"Heatmap (bin_size={bin_size}) - {len(peaks)} peaks")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        
        plt.tight_layout()
        self._save_fig(fig, f"votes_{title.replace(' ', '_')}")
    
    def show_peeling_step(self, img: np.ndarray, bbox: np.ndarray, 
                          inliers: int, iteration: int, book_id: str):
        """Mostra ogni iterazione del peeling."""
        if not self.enabled:
            return
        
        vis = img.copy()
        cv2.polylines(vis, [np.int32(bbox)], True, (0, 255, 0), 3)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Peeling iter {iteration}: {book_id} ({inliers} inliers)")
        ax.axis('off')
        
        plt.tight_layout()
        self._save_fig(fig, f"peel_{book_id}_iter{iteration}")
    
    def show_nms_comparison(self, img: np.ndarray, 
                            before_nms: List[dict], after_nms: List[dict]):
        """Mostra prima e dopo NMS."""
        if not self.enabled:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Prima di NMS
        vis1 = img.copy()
        for r in before_nms:
            cv2.polylines(vis1, [np.int32(r['bbox'])], True, (0, 0, 255), 2)
        axes[0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"PRIMA di NMS: {len(before_nms)} detections")
        axes[0].axis('off')
        
        # Dopo NMS
        vis2 = img.copy()
        colors = plt.cm.tab10.colors
        for i, r in enumerate(after_nms):
            color = tuple(int(c * 255) for c in colors[i % 10][:3])
            cv2.polylines(vis2, [np.int32(r['bbox'])], True, color, 3)
            pt = tuple(np.int32(r['bbox']).reshape(-1, 2)[0])
            cv2.putText(vis2, f"{r['book_id']}", pt, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        axes[1].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"DOPO NMS: {len(after_nms)} detections")
        axes[1].axis('off')
        
        plt.tight_layout()
        self._save_fig(fig, "nms_comparison")
    
    def show_full_summary(self, img: np.ndarray, model_stats: Dict[str, int],
                          match_stats: Dict[str, int], results: List[dict]):
        """Mostra summary completo dell'analisi."""
        if not self.enabled:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Immagine con risultati
        vis = img.copy()
        colors = plt.cm.tab10.colors
        for i, r in enumerate(results):
            color = tuple(int(c * 255) for c in colors[i % 10][:3])
            cv2.polylines(vis, [np.int32(r['bbox'])], True, color, 3)
            pt = tuple(np.int32(r['bbox']).reshape(-1, 2)[0])
            cv2.putText(vis, f"{r['book_id']} ({r['score']})", pt,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Risultato Finale: {len(results)} libri")
        axes[0].axis('off')
        
        # Statistiche modelli (keypoints per modello)
        if model_stats:
            models = list(model_stats.keys())
            kps = list(model_stats.values())
            axes[1].barh(models, kps, color='steelblue')
            axes[1].set_xlabel('Keypoints')
            axes[1].set_title('Keypoints per Modello')
            axes[1].invert_yaxis()
        
        # Statistiche match
        if match_stats:
            models = list(match_stats.keys())
            matches = list(match_stats.values())
            
            # Evidenzia modelli che hanno prodotto detection
            detected_models = {r['book_id'] for r in results}
            colors_bar = ['green' if m in detected_models else 'gray' for m in models]
            
            axes[2].barh(models, matches, color=colors_bar)
            axes[2].set_xlabel('Matches')
            axes[2].set_title('Matches per Modello\n(verde=detected)')
            axes[2].invert_yaxis()
        
        plt.tight_layout()
        self._save_fig(fig, "summary")


# ===========================================================================
#  PREPROCESSING
# ===========================================================================

class Preprocessor:
    """CLAHE + Unsharp mask (senza morfologia - riduce keypoints del 30%)."""
    
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
        )

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        
        # CLAHE - migliora contrasto locale
        l = self._clahe.apply(l)
        
        # Unsharp mask - aumenta nitidezza dei bordi
        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=2.0)
        alpha = self._cfg.sharpen_amount
        l = cv2.addWeighted(l, 1.0 + alpha, blur, -alpha, 0)
        
        return l


# ===========================================================================
#  FEATURE INDEX
# ===========================================================================

class FeatureIndex:
    """SIFT + FLANN matching."""
    
    def __init__(self, cfg: Config, diag: Diagnostics) -> None:
        self.cfg = cfg
        self.diag = diag
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
        # Per debug: memorizza immagini e keypoints modelli
        self._model_images: Dict[str, np.ndarray] = {}
        self._model_kps: Dict[str, List[cv2.KeyPoint]] = {}

    def extract(self, gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        kp, des = self._sift.detectAndCompute(gray, None)
        if des is None:
            return [], None
        des = des.astype(np.float32)
        if self.cfg.root_sift:
            des /= des.sum(axis=1, keepdims=True) + 1e-7
            np.sqrt(des, out=des)
        return kp, des

    def add(self, book_id: str, des: np.ndarray, img: np.ndarray, kp: List[cv2.KeyPoint]) -> None:
        self._matcher.add([des])
        self._ids.append(book_id)
        self._model_images[book_id] = img
        self._model_kps[book_id] = kp

    def build(self) -> None:
        self._matcher.train()

    def match(self, des_scene: np.ndarray, kp_scene: List[cv2.KeyPoint],
              model_kps: Dict[str, List[cv2.KeyPoint]]) -> Dict[str, list]:
        """Match con diagnostica dettagliata."""
        if des_scene is None or not self._ids:
            return {}
        
        raw = self._matcher.knnMatch(des_scene, k=2)
        
        # Conta match pre-Lowe
        pre_lowe = len([p for p in raw if len(p) >= 2])
        
        # Applica Lowe ratio
        groups: Dict[str, list] = defaultdict(list)
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg.lowe_ratio * n.distance:
                groups[self._ids[m.imgIdx]].append(m)
        
        post_lowe = sum(len(ms) for ms in groups.values())
        
        # Scale-ratio filter
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
        
        post_scale = sum(len(ms) for ms in filtered.values())
        
        # Log statistiche
        print(f"    [Match] Pre-Lowe: {pre_lowe}, Post-Lowe: {post_lowe}, Post-Scale: {post_scale}")
        
        return filtered


# ===========================================================================
#  STAR MODEL CON DEBUG
# ===========================================================================

class StarModel:
    """GHT + Peeling con diagnostica."""
    
    def __init__(self, cfg: Config, index: FeatureIndex, diag: Diagnostics) -> None:
        self.cfg = cfg
        self.index = index
        self.diag = diag
        self._tpl: Dict[str, dict] = {}

    def register(self, book_id: str, kp: List[cv2.KeyPoint], shape: Tuple[int, ...]) -> None:
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
        self._tpl[book_id] = {"kp": kp, "vecs": vecs, "shape": shape}

    def detect(self, kp_scene: List[cv2.KeyPoint], des_scene: Optional[np.ndarray],
               scene_shape: Tuple[int, ...], scene_img: np.ndarray = None) -> List[dict]:
        if des_scene is None:
            return []
        
        model_kps = {bid: tpl["kp"] for bid, tpl in self._tpl.items()}
        all_matches = self.index.match(des_scene, kp_scene, model_kps)
        
        # Log match stats
        match_stats = {bid: len(ms) for bid, ms in all_matches.items() if len(ms) > 0}
        self.diag.log('match_stats', match_stats)
        
        cands: List[dict] = []
        
        for bid, matches in all_matches.items():
            tpl = self._tpl[bid]
            
            if len(matches) < self.cfg.min_votes:
                continue
            
            print(f"    [GHT] {bid}: {len(matches)} matches")
            
            # Calcola voti e peaks
            peaks, all_votes, bins = self._vote_debug(matches, kp_scene, tpl, scene_shape)
            
            # Mostra heatmap voti
            if len(all_votes) > 0:
                peak_coords = [(bx, by) for bx, by in bins.keys() if len(bins[(bx,by)]) >= self.cfg.min_votes]
                self.diag.show_votes_heatmap(all_votes, scene_shape, self.cfg.bin_size, 
                                             peak_coords, f"{bid}")
            
            for peak in peaks:
                extracted = self._extract_debug(bid, peak, tpl, kp_scene, scene_shape, scene_img)
                cands.extend(extracted)
        
        # NMS con debug
        self.diag.log('before_nms', cands.copy())
        kept = self._nms(cands)
        self.diag.log('after_nms', kept)
        
        if scene_img is not None and cands:
            self.diag.show_nms_comparison(scene_img, cands, kept)
        
        return kept

    def _vote_debug(self, matches: list, kp_scene: List[cv2.KeyPoint],
                    tpl: dict, scene_shape: Tuple[int, ...]) -> Tuple[List[list], np.ndarray, dict]:
        """Voting con clustering spaziale invece di binning rigido."""
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
            return [], valid_votes, {}
        
        # Usa clustering gerarchico per raggruppare voti vicini
        # cluster_dist = distanza max tra elementi nello stesso cluster
        cluster_dist = self.cfg.bin_size * 2  # Raggio ragionevole
        
        clusters = fclusterdata(valid_votes, t=cluster_dist, criterion='distance', method='single')
        
        # Raggruppa match per cluster
        groups: Dict[int, list] = defaultdict(list)
        for i, c in enumerate(clusters):
            groups[c].append(valid_matches[i])
        
        # Filtra per min_votes e ordina per dimensione
        peaks = [g for g in groups.values() if len(g) >= self.cfg.min_votes]
        peaks.sort(key=len, reverse=True)
        
        # Per compatibilità con heatmap, crea anche bins
        bs = self.cfg.bin_size
        bins: Dict[Tuple[int, int], list] = defaultdict(list)
        for i, v in enumerate(valid_votes):
            bx, by = int(v[0] // bs), int(v[1] // bs)
            bins[(bx, by)].append(valid_matches[i])
        
        # Log clustering info
        print(f"      Clusters trovati: {len(peaks)}, match per cluster: {[len(g) for g in peaks[:5]]}")
        print(f"      Top bins: {[(k, len(v)) for k, v in sorted(bins.items(), key=lambda x: len(x[1]), reverse=True)[:5]]}")
        
        return peaks, valid_votes, bins
        
        return peaks, votes[valid], bins

    def _merge(self, bins: Dict[Tuple[int, int], list]) -> List[list]:
        """Merge con merge_radius configurabile."""
        r = self.cfg.merge_radius
        keys = sorted(bins, key=lambda b: len(bins[b]), reverse=True)
        used: set = set()
        groups: List[list] = []
        
        for b in keys:
            if b in used:
                continue
            g = list(bins[b])
            used.add(b)
            
            if r > 0:  # Solo se merge_radius > 0
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nb = (b[0] + dx, b[1] + dy)
                        if nb != b and nb not in used and nb in bins:
                            g.extend(bins[nb])
                            used.add(nb)
            
            if len(g) >= self.cfg.min_votes:
                groups.append(g)
        
        groups.sort(key=len, reverse=True)
        print(f"      Peaks trovati: {len(groups)}, voti per peak: {[len(g) for g in groups[:5]]}")
        return groups

    def _extract_debug(self, bid: str, matches: list, tpl: dict,
                       kp_scene: list, scene_shape: Tuple[int, ...],
                       scene_img: np.ndarray = None) -> List[dict]:
        """Peeling con visualizzazione."""
        found: List[dict] = []
        pool = list(matches)
        iteration = 0
        
        while len(pool) >= self.cfg.min_inliers:
            r = self._verify(bid, pool, tpl, kp_scene, scene_shape)
            if r is None:
                print(f"        Peeling iter {iteration}: FAILED (no valid transform)")
                break
            
            found.append(r)
            print(f"        Peeling iter {iteration}: {bid} con {r['score']} inliers")
            
            # Visualizza
            if scene_img is not None:
                self.diag.show_peeling_step(scene_img, r['bbox'], r['score'], iteration, bid)
            
            pool = self._strip(pool, tpl, kp_scene, r["H"])
            iteration += 1
            
            if iteration >= 10:  # Safety limit
                break
        
        return found

    def _strip(self, matches: list, tpl: dict, kp_scene: list, H: np.ndarray) -> list:
        src = np.float32([tpl["kp"][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_scene[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        if H.shape[0] == 2:
            proj = cv2.transform(src, H).reshape(-1, 2)
        else:
            proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        return [m for m, e in zip(matches, err) if e >= self.cfg.ransac_thresh * 1.5]

    def _verify(self, bid: str, matches: list, tpl: dict,
                kp_scene: list, scene_shape: Tuple[int, ...]) -> Optional[dict]:
        src = np.float32([tpl["kp"][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_scene[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        use_affine = len(matches) < self.cfg.affine_thresh
        method = "Affine" if use_affine else "Homography"

        if use_affine:
            M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                   ransacReprojThreshold=self.cfg.ransac_thresh)
            if M is None or mask is None:
                print(f"          [REJECT] {method} estimation failed")
                return None
            H = M
        else:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.cfg.ransac_thresh)
            if H is None or mask is None:
                print(f"          [REJECT] Homography estimation failed")
                return None

        n_inliers = int(mask.ravel().sum())
        min_inl = self.cfg.min_inliers if use_affine else self.cfg.min_inliers_h
        if n_inliers < min_inl:
            print(f"          [REJECT] Inliers={n_inliers} < min={min_inl}")
            return None

        det = np.linalg.det(H[:2, :2])
        if det < self.cfg.min_det or det > self.cfg.max_det:
            print(f"          [REJECT] Determinant={det:.4f} out of range [{self.cfg.min_det}, {self.cfg.max_det}]")
            return None

        h, w = tpl["shape"][:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        if use_affine:
            bbox = cv2.transform(corners, H)
        else:
            bbox = cv2.perspectiveTransform(corners, H)

        if not cv2.isContourConvex(bbox.astype(np.int32)):
            print(f"          [REJECT] Bbox not convex")
            return None

        area = cv2.contourArea(bbox)
        scene_area = scene_shape[0] * scene_shape[1]
        if area < self.cfg.min_area or area > scene_area * self.cfg.max_area_frac:
            print(f"          [REJECT] Area={area:.0f} out of range [{self.cfg.min_area}, {scene_area * self.cfg.max_area_frac:.0f}]")
            return None
        
        # Score minimo basato sulla complessità del modello
        model_kp_count = len(tpl["kp"])
        score_threshold = max(3, int(model_kp_count * 0.003))  # 0.3% dei keypoints del modello
        if n_inliers < score_threshold:
            print(f"          [REJECT] Score={n_inliers} < threshold={score_threshold}")
            return None
        
        print(f"          [ACCEPT] inliers={n_inliers}, det={det:.3f}, area={area:.0f}")

        return {"book_id": bid, "score": n_inliers, "bbox": bbox, "H": H}

    def _nms(self, cands: List[dict]) -> List[dict]:
        if not cands:
            return []
        print(f"  [NMS] Input candidati: {len(cands)}")
        cands.sort(key=lambda c: c["score"], reverse=True)
        kept: List[dict] = []
        for i, c in enumerate(cands):
            cp = c["bbox"].reshape(4, 2).astype(np.float32)
            dup = False
            for k in kept:
                kp_ = k["bbox"].reshape(4, 2).astype(np.float32)
                thresh = self.cfg.iou_same if c["book_id"] == k["book_id"] else self.cfg.iou_cross
                iou_val = self._iou(cp, kp_)
                if iou_val > thresh:
                    print(f"    [NMS] Candidato {i} ({c['book_id']}, score={c['score']}) SCARTATO: IoU={iou_val:.3f} > {thresh}")
                    dup = True
                    break
            if not dup:
                print(f"    [NMS] Candidato {i} ({c['book_id']}, score={c['score']}) KEPT")
                kept.append(c)
        return kept

    @staticmethod
    def _iou(p1: np.ndarray, p2: np.ndarray) -> float:
        ok, inter = cv2.intersectConvexConvex(p1, p2)
        if not ok or inter is None:
            return 0.0
        ia = cv2.contourArea(inter)
        return ia / (cv2.contourArea(p1) + cv2.contourArea(p2) - ia + 1e-7)


# ===========================================================================
#  PIPELINE ORCHESTRATOR
# ===========================================================================

class BookRecognizer:
    """Pipeline completa con diagnostica."""
    
    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.diag = Diagnostics(enabled=self.cfg.debug)
        self.prep = Preprocessor(self.cfg)
        self.index = FeatureIndex(self.cfg, self.diag)
        self.star = StarModel(self.cfg, self.index, self.diag)
        self.model_stats: Dict[str, int] = {}

    def load_library(self, path: str) -> None:
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
            self.index.add(bid, des, enhanced, kp)
            self.model_stats[bid] = len(kp)
            n += 1
            print(f"  {bid}: {len(kp)} keypoints")
        
        self.index.build()
        print(f"{n} models indexed.\n")
        self.diag.log('model_stats', self.model_stats)

    def run(self, scenes_path: str, specific_scene: str = None) -> None:
        """Esegui su scene (o una specifica per debug)."""
        files = sorted(
            f for f in os.listdir(scenes_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        
        if specific_scene:
            files = [f for f in files if specific_scene in f]
        
        print(f"Processing {len(files)} scenes ...")
        total = 0
        
        for f in files:
            print(f"\n{'='*60}")
            print(f"SCENE: {f}")
            print(f"{'='*60}")
            
            img = cv2.imread(os.path.join(scenes_path, f))
            if img is None:
                continue
            
            # Preprocessing con debug
            enhanced = self.prep(img)
            if self.cfg.debug:
                self.diag.show_preprocessing(img, enhanced, f)
            
            # Estrazione keypoints
            kp, des = self.index.extract(enhanced)
            print(f"  Scene keypoints: {len(kp)}")
            
            if self.cfg.debug:
                self.diag.show_keypoints(enhanced, kp, f)
            
            # Detection
            results = self.star.detect(kp, des, enhanced.shape, img)
            
            # Summary finale
            match_stats = self.diag.data.get('match_stats', {})
            self.diag.show_full_summary(img, self.model_stats, match_stats, results)
            
            # Visualizza risultato
            self._draw(img, results, f)
            total += len(results)
        
        print(f"\n{'='*60}")
        print(f"TOTALE: {total} libri detectati su {len(files)} scene")
        print(f"{'='*60}")

    def run_with_eval(self, scenes_path: str) -> Dict[str, List[dict]]:
        """Esegui su tutte le scene e ritorna risultati per valutazione."""
        files = sorted(
            f for f in os.listdir(scenes_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        
        print(f"Processing {len(files)} scenes ...")
        all_results: Dict[str, List[dict]] = {}
        
        for f in files:
            print(f"\n{'='*60}")
            print(f"SCENE: {f}")
            print(f"{'='*60}")
            
            img = cv2.imread(os.path.join(scenes_path, f))
            if img is None:
                continue
            
            # Preprocessing
            enhanced = self.prep(img)
            
            # Estrazione keypoints
            kp, des = self.index.extract(enhanced)
            print(f"  Scene keypoints: {len(kp)}")
            
            # Detection
            results = self.star.detect(kp, des, enhanced.shape, img)
            
            # Salva risultati
            all_results[f] = results
            
            # Visualizza risultato (salva su file)
            self._draw(img, results, f)
        
        total = sum(len(r) for r in all_results.values())
        print(f"\n{'='*60}")
        print(f"TOTALE: {total} libri detectati su {len(files)} scene")
        print(f"{'='*60}")
        
        return all_results

    @staticmethod
    def _draw(img: np.ndarray, results: List[dict], name: str, output_dir: str = "debug_output") -> None:
        """Salva risultato su file invece di mostrarlo."""
        os.makedirs(output_dir, exist_ok=True)
        
        vis = img.copy()
        colors = plt.cm.tab10.colors
        
        for i, r in enumerate(results):
            bbox = np.int32(r["bbox"])
            color = tuple(int(c * 255) for c in colors[i % 10][:3])
            
            overlay = vis.copy()
            cv2.fillConvexPoly(overlay, bbox, color)
            cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
            cv2.polylines(vis, [bbox], True, color, 3)
            
            label = f"{r['book_id']} ({r['score']})"
            pts = bbox.reshape(-1, 2)
            x, y = pts[pts[:, 1].argmin()]
            cv2.putText(vis, label, (x, max(y - 5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Salva su file
        out_path = os.path.join(output_dir, f"result_{name}")
        cv2.imwrite(out_path, vis)
        print(f"  [Saved] {out_path}")
        
        for r in results:
            print(f"  -> {r['book_id']} ({r['score']} inliers)")


# ===========================================================================
#  GROUND TRUTH & METRICHE
# ===========================================================================

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
    """Calcola precision, recall e F1 per ogni scena e totale."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print(f"\n{'='*70}")
    print("VALUTAZIONE vs GROUND TRUTH")
    print(f"{'='*70}")
    
    for scene_name, results in all_results.items():
        # Estrai scene key (es. "scene_10" da "scene_10.jpg")
        scene_key = scene_name.replace(".jpg", "").replace(".png", "")
        
        gt = GROUND_TRUTH.get(scene_key, [])
        detected = [r['book_id'] for r in results]
        
        # Conta matches (greedy matching)
        gt_remaining = list(gt)
        tp = 0
        for d in detected:
            if d in gt_remaining:
                tp += 1
                gt_remaining.remove(d)
        
        fp = len(detected) - tp
        fn = len(gt) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if gt or detected:
            status = "✓" if fp == 0 and fn == 0 else "✗"
            print(f"  {status} {scene_key}: GT={len(gt)}, Det={len(detected)}, TP={tp}, FP={fp}, FN={fn}")
            if fp > 0 or fn > 0:
                print(f"      GT: {gt}")
                print(f"      Detected: {detected}")
    
    # Metriche globali
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"TOTALE: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"{'='*70}")
    
    return {"precision": precision, "recall": recall, "f1": f1}


# ===========================================================================
#  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    cfg = Config(
        debug=False,
        
        # SIFT
        contrast_threshold=0.01,
        edge_threshold=15,
        
        # GHT 
        bin_size=18,
        merge_radius=0,
        min_votes=3,
        
        # Matching
        lowe_ratio=0.78,    # Stretto per meno FP
        scale_sigma=3.5,
        
        # Verifica geometrica
        min_inliers=3,
        min_inliers_h=5,
        affine_thresh=10,
        min_area=450,
        min_det=0.025,
        max_det=40.0,
        
        # NMS
        iou_same=0.30,
        iou_cross=0.35,
    )
    
    rec = BookRecognizer(cfg)
    rec.load_library("./dataset/models/")
    
    # Esegui e raccogli risultati
    all_results = rec.run_with_eval("./dataset/scenes/")
    
    # Valuta vs ground truth
    metrics = evaluate_results(all_results)
