"""
NCC Verification Analysis
=========================
Analyses NCC scores of detections from the SIFT+FLANN pipeline
to see if false positives can be distinguished from true positives.
"""

import os
import cv2
import numpy as np
from assignement_debug import Config, Preprocessor, FeatureIndex, StarModel, BookRecognizer, GROUND_TRUTH


def compute_ncc(model_gray: np.ndarray, scene_gray: np.ndarray, H: np.ndarray) -> float:
    """Warp model into scene space using H and compute NCC in the overlap region."""
    h_m, w_m = model_gray.shape[:2]
    h_s, w_s = scene_gray.shape[:2]
    
    # Handle both affine (2x3) and homography (3x3) transforms
    if H.shape[0] == 2:
        warped = cv2.warpAffine(model_gray, H, (w_s, h_s),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ones_like(model_gray) * 255, H, (w_s, h_s),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        warped = cv2.warpPerspective(model_gray, H, (w_s, h_s),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.ones_like(model_gray) * 255, H, (w_s, h_s),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = mask > 128
    
    if mask.sum() < 100:
        return 0.0
    
    # Extract pixels
    w_pix = warped[mask].astype(np.float64)
    s_pix = scene_gray[mask].astype(np.float64)
    
    # Normalize
    w_pix -= w_pix.mean()
    s_pix -= s_pix.mean()
    
    w_std = w_pix.std()
    s_std = s_pix.std()
    
    if w_std < 1e-8 or s_std < 1e-8:
        return 0.0
    
    ncc = np.dot(w_pix, s_pix) / (w_std * s_std * len(w_pix))
    return float(ncc)


def compute_ssim_score(model_gray: np.ndarray, scene_gray: np.ndarray, H: np.ndarray) -> float:
    """Compute structural similarity between warped model and scene."""
    h_s, w_s = scene_gray.shape[:2]
    
    if H.shape[0] == 2:
        warped = cv2.warpAffine(model_gray, H, (w_s, h_s),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.ones_like(model_gray) * 255, H, (w_s, h_s),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        warped = cv2.warpPerspective(model_gray, H, (w_s, h_s),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.ones_like(model_gray) * 255, H, (w_s, h_s),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Compute SSIM in the masked region
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    w = warped.astype(np.float64)
    s = scene_gray.astype(np.float64)
    m = (mask > 128).astype(np.float64)
    
    n = m.sum()
    if n < 100:
        return 0.0
    
    mu_w = (w * m).sum() / n
    mu_s = (s * m).sum() / n
    
    sig_w = np.sqrt(((w - mu_w)**2 * m).sum() / n)
    sig_s = np.sqrt(((s - mu_s)**2 * m).sum() / n)
    sig_ws = ((w - mu_w) * (s - mu_s) * m).sum() / n
    
    ssim = ((2 * mu_w * mu_s + C1) * (2 * sig_ws + C2)) / \
           ((mu_w**2 + mu_s**2 + C1) * (sig_w**2 + sig_s**2 + C2))
    
    return float(ssim)


def main():
    cfg = Config(
        debug=False,
        contrast_threshold=0.01,
        edge_threshold=15,
        bin_size=18,
        merge_radius=0,
        min_votes=3,
        lowe_ratio=0.78,
        scale_sigma=3.5,
        min_inliers=3,
        min_inliers_h=5,
        affine_thresh=10,
        min_area=450,
        min_det=0.025,
        max_det=40.0,
        iou_same=0.30,
        iou_cross=0.35,
    )
    
    rec = BookRecognizer(cfg)
    rec.load_library("./dataset/models/")
    
    # Run detection
    files = sorted(f for f in os.listdir("./dataset/scenes/")
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    
    prep = Preprocessor(cfg)
    
    tp_nccs = []
    fp_nccs = []
    tp_ssims = []
    fp_ssims = []
    
    for f in files:
        scene_key = f.replace(".jpg", "").replace(".png", "")
        gt = GROUND_TRUTH.get(scene_key, [])
        
        img = cv2.imread(os.path.join("./dataset/scenes/", f))
        if img is None:
            continue
        
        enhanced = prep(img)
        kp, des = rec.index.extract(enhanced)
        results = rec.star.detect(kp, des, enhanced.shape, img)
        
        if not results:
            continue
            
        # Classify TP vs FP
        gt_rem = list(gt)
        for r in sorted(results, key=lambda x: -x["score"]):
            bid = r["book_id"]
            H = r["H"]
            
            # Get model image
            model_img = rec.index._model_images.get(bid)
            if model_img is None:
                continue
            
            ncc = compute_ncc(model_img, enhanced, H)
            ssim = compute_ssim_score(model_img, enhanced, H)
            
            if bid in gt_rem:
                gt_rem.remove(bid)
                tp_nccs.append(ncc)
                tp_ssims.append(ssim)
                label = "TP"
            else:
                fp_nccs.append(ncc)
                fp_ssims.append(ssim)
                label = "FP"
            
            print(f"  {scene_key}: {bid} ({r['score']} inliers) → {label} | NCC={ncc:.3f} SSIM={ssim:.3f}")
    
    print(f"\n{'='*60}")
    print(f"NCC Analysis")
    print(f"{'='*60}")
    if tp_nccs:
        print(f"TP NCC:  min={min(tp_nccs):.3f}, max={max(tp_nccs):.3f}, "
              f"mean={np.mean(tp_nccs):.3f}, median={np.median(tp_nccs):.3f}")
    if fp_nccs:
        print(f"FP NCC:  min={min(fp_nccs):.3f}, max={max(fp_nccs):.3f}, "
              f"mean={np.mean(fp_nccs):.3f}, median={np.median(fp_nccs):.3f}")
    
    print(f"\nSSIM Analysis")
    if tp_ssims:
        print(f"TP SSIM: min={min(tp_ssims):.3f}, max={max(tp_ssims):.3f}, "
              f"mean={np.mean(tp_ssims):.3f}, median={np.median(tp_ssims):.3f}")
    if fp_ssims:
        print(f"FP SSIM: min={min(fp_ssims):.3f}, max={max(fp_ssims):.3f}, "
              f"mean={np.mean(fp_ssims):.3f}, median={np.median(fp_ssims):.3f}")
    
    # Test NCC thresholds
    print(f"\nNCC Threshold Analysis:")
    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        tp_kept = sum(1 for x in tp_nccs if x >= thr)
        fp_removed = sum(1 for x in fp_nccs if x < thr)
        tp_lost = len(tp_nccs) - tp_kept
        fp_kept = len(fp_nccs) - fp_removed
        new_tp = tp_kept
        new_fp = fp_kept
        new_fn = 57 - new_tp
        p = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
        r = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  NCC>={thr:.2f}: TP_kept={tp_kept}/{len(tp_nccs)}, "
              f"FP_removed={fp_removed}/{len(fp_nccs)}, "
              f"P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    
    # Test SSIM thresholds
    print(f"\nSSIM Threshold Analysis:")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        tp_kept = sum(1 for x in tp_ssims if x >= thr)
        fp_removed = sum(1 for x in fp_ssims if x < thr)
        tp_lost = len(tp_ssims) - tp_kept
        fp_kept = len(fp_ssims) - fp_removed
        new_tp = tp_kept
        new_fp = fp_kept
        new_fn = 57 - new_tp
        p = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
        r = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  SSIM>={thr:.2f}: TP_kept={tp_kept}/{len(tp_ssims)}, "
              f"FP_removed={fp_removed}/{len(fp_ssims)}, "
              f"P={p:.3f}, R={r:.3f}, F1={f1:.3f}")


if __name__ == "__main__":
    main()
