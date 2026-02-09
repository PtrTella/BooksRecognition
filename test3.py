import cv2
import numpy as np
from scipy.signal import find_peaks


class LibrarySegmenter:
    def __init__(
        self,
        sobel_ksize: int = 3,
        smooth_kernel: int = 21,
        peak_prominence: float = 0.2,
        min_book_width_px: int = 30,
        peak_distance_px: int = 30,
        vertical_kernel_height: int = 25,
        min_vertical_density: float = 0.15,
        debug: bool = False,
    ):
        """
        Parametri principali da tarare:
            smooth_kernel        : quanto lisciare il profilo (dispari)
            peak_prominence      : quanto "importante" deve essere un picco (0..1)
            min_book_width_px    : larghezza minima di un libro
            peak_distance_px     : distanza minima tra due bordi
            vertical_kernel_height: altezza kernel per tenere solo bordi verticali lunghi
            min_vertical_density : soglia di densità verticale per accettare un bordo
        """
        self.sobel_ksize = sobel_ksize
        self.smooth_kernel = smooth_kernel
        self.peak_prominence = peak_prominence
        self.min_book_width_px = min_book_width_px
        self.peak_distance_px = peak_distance_px
        self.vertical_kernel_height = vertical_kernel_height
        self.min_vertical_density = min_vertical_density
        self.debug = debug

    # ------------ API principale ------------

    def segment_shelf(self, shelf_bgr):
        """
        Input: immagine BGR di un singolo ripiano.
        Output:
            boxes: lista di [x1, y1, x2, y2]
            (opzionale) debug_dict se debug=True
        """
        gray = self._to_gray_equalized(shelf_bgr)
        edges_vertical = self._vertical_edge_mask(gray)
        profile = self._column_profile(edges_vertical)
        profile_smooth, profile_norm = self._smooth_and_normalize(profile)
        borders_raw, peaks_props = self._detect_profile_peaks(profile_norm)
        borders_filtered = self._filter_borders_by_vertical_density(
            borders_raw, edges_vertical
        )
        refined_borders = self._refine_borders_by_profile(
            borders_filtered, profile_norm
        )

        h, w = gray.shape[:2]
        boxes = self._borders_to_boxes(refined_borders, w, h)

        if not self.debug:
            return boxes

        debug_dict = {
            "gray": gray,
            "edges_vertical": edges_vertical,
            "profile": profile,
            "profile_norm": profile_norm,
            "borders_raw": borders_raw,
            "borders_filtered": borders_filtered,
            "borders_refined": refined_borders,
            "peaks_properties": peaks_props,
        }
        return boxes, debug_dict

    # ------------ Step interni ------------

    def _to_gray_equalized(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def _vertical_edge_mask(self, gray):
        """Restituisce una mappa binaria dei bordi verticali lunghi."""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        abs_grad_x = np.abs(grad_x)
        grad_uint8 = cv2.convertScaleAbs(abs_grad_x)

        # threshold Otsu sui gradienti
        _, edges = cv2.threshold(
            grad_uint8, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # apertura verticale: rimuove segmenti corti (es. lettere)
        kernel_vert = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, self.vertical_kernel_height)
        )
        edges_vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_vert)

        # leggera dilatazione per connettere eventuali buchi
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_vert = cv2.dilate(edges_vert, kernel_dil, iterations=1)

        return edges_vert

    def _column_profile(self, edges_vertical):
        """Profilo per colonna: quanta 'linea verticale' c'è in ogni x."""
        h, w = edges_vertical.shape[:2]
        # mean (0..255) -> normalizzato dopo
        col_profile = edges_vertical.mean(axis=0).astype(np.float64)
        return col_profile

    def _smooth_and_normalize(self, profile):
        if self.smooth_kernel > 1:
            k = self.smooth_kernel
            kernel = np.ones(k, dtype=np.float64) / float(k)
            smooth = np.convolve(profile, kernel, mode="same")
        else:
            smooth = profile.copy()

        if smooth.max() > 0:
            norm = smooth / smooth.max()
        else:
            norm = smooth
        return smooth, norm

    def _detect_profile_peaks(self, profile_norm):
        """Trova picchi nel profilo normalizzato."""
        peaks, props = find_peaks(
            profile_norm,
            prominence=self.peak_prominence,
            distance=self.peak_distance_px,
        )
        return peaks.tolist(), props

    def _filter_borders_by_vertical_density(self, borders, edges_vertical):
        """
        Scarta i bordi per cui, in una fascia centrale verticale,
        la densità di pixel-bordo è troppo bassa (tipico delle scritte).
        """
        h, w = edges_vertical.shape[:2]
        mid_top = int(0.2 * h)
        mid_bot = int(0.8 * h)

        filtered = []
        for x in sorted(borders):
            if x < 0 or x >= w:
                continue
            col = edges_vertical[mid_top:mid_bot, x]
            density = col.mean() / 255.0  # 0..1
            if density >= self.min_vertical_density:
                filtered.append(x)

        # inoltre imponi una distanza minima fra bordi
        merged = []
        for x in sorted(filtered):
            if not merged:
                merged.append(x)
            else:
                if x - merged[-1] < self.min_book_width_px:
                    # tieni quello la cui densità locale è maggiore (già in filtered, quindi semplifico)
                    # qui faccio una media semplice; puoi raffinare usando profile_norm
                    merged[-1] = int((merged[-1] + x) / 2)
                else:
                    merged.append(x)
        return merged

    def _refine_borders_by_profile(self, borders, profile_norm, window_half_width=3):
        """Raffina ogni bordo scegliendo il massimo del profilo in una piccola finestra."""
        w = len(profile_norm)
        refined = []
        for x in sorted(borders):
            x0 = max(0, x - window_half_width)
            x1 = min(w - 1, x + window_half_width)
            local = profile_norm[x0:x1 + 1]
            best = int(np.argmax(local))
            refined.append(x0 + best)
        return sorted(set(refined))

    def _borders_to_boxes(self, borders, width, height):
        """Converte bordi x in bounding box [x1, y1, x2, y2]."""
        all_borders = [0] + sorted(borders) + [width]
        boxes = []
        for i in range(len(all_borders) - 1):
            x1 = all_borders[i]
            x2 = all_borders[i + 1]
            if x2 - x1 < self.min_book_width_px:
                continue
            boxes.append([x1, 0, x2, height])
        return boxes


if __name__ == "__main__!!":
    img = cv2.imread("./dataset/scenes/scene_0.jpg")

    segmenter = LibrarySegmenter(
        smooth_kernel=11,
        peak_prominence=0.1,
        min_book_width_px=10,
        peak_distance_px=15,
        vertical_kernel_height=35,
        min_vertical_density=0.25,
        debug=True,
    )

    boxes, debug_info = segmenter.segment_shelf(img)

    # show all scene from folder "./dataset/scenes/" in one window with detected boxes
    import os
    scene_folder = "./dataset/scenes/"
    scene_images = [f for f in os.listdir(scene_folder) if f.endswith(".jpg")]
    for scene_file in scene_images:
        img = cv2.imread(os.path.join(scene_folder, scene_file))
        boxes, _ = segmenter.segment_shelf(img)
        vis = img.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow(f"Scene: {scene_file}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class BookIdentifier:
    """
    Identificazione libri tramite:
    - descrittori locali (SIFT o ORB)
    - istogramma di colore HSV
    e (opzionale) raffinamento bbox via omografia SIFT.
    """

    def __init__(
        self,
        use_sift: bool = True,
        nfeatures_orb: int = 1000,
        color_hist_bins: Tuple[int, int, int] = (16, 16, 4),
        ratio_test: float = 0.75,
        min_good_matches: int = 20,
        min_color_similarity: float = 0.6,
        alpha_local: float = 1.0,
        beta_color: float = 0.5,
        homography_ransac_thresh: float = 5.0,
    ):
        self.detector, self.norm_type = self._create_feature_detector(
            use_sift=use_sift, nfeatures_orb=nfeatures_orb
        )
        self.bins_h, self.bins_s, self.bins_v = color_hist_bins
        self.ratio_test = ratio_test
        self.min_good_matches = min_good_matches
        self.min_color_similarity = min_color_similarity
        self.alpha_local = alpha_local
        self.beta_color = beta_color
        self.homography_ransac_thresh = homography_ransac_thresh

        # db: book_id -> dict con kp, des, hist, img, template_bbox
        # template_bbox: [0, 0, W, H] di solito, ma puoi salvare anche bbox interne
        self.templates: Dict[str, dict] = {}

    # ---------- init detector ----------

    def _create_feature_detector(self, use_sift: bool, nfeatures_orb: int):
        if use_sift:
            try:
                sift = cv2.SIFT_create()
                return sift, cv2.NORM_L2
            except Exception:
                orb = cv2.ORB_create(nfeatures=nfeatures_orb)
                return orb, cv2.NORM_HAMMING
        else:
            orb = cv2.ORB_create(nfeatures=nfeatures_orb)
            return orb, cv2.NORM_HAMMING

    # ---------- feature + colore ----------

    def _compute_features(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des

    def _compute_color_hist(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self.bins_h, self.bins_s, self.bins_v],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist

    # ---------- gestione template ----------

    def add_template(self, book_id: str, img_bgr):
        """
        Aggiunge un libro al database dei template.
        Il template di default usa tutta l'immagine come bbox.
        """
        kp, des = self._compute_features(img_bgr)
        if des is None or len(kp) == 0:
            raise ValueError(f"Nessuna feature trovata per il template {book_id}")

        hist = self._compute_color_hist(img_bgr)
        h, w = img_bgr.shape[:2]

        self.templates[book_id] = {
            "kp": kp,
            "des": des,
            "hist": hist,
            "img": img_bgr,
            "bbox": [0, 0, w, h],
        }

    # ---------- matching semplice (solo ID) ----------

    def match_book(
        self,
        book_roi_bgr,
    ) -> Tuple[Optional[str], float]:
        """
        Matching standard: ritorna solo l'ID del libro e il punteggio.
        """
        best_id, best_score, _ = self._match_and_homography(book_roi_bgr)
        return best_id, best_score

    # ---------- matching + omografia (bbox SIFT "trasposta") ----------

    def match_book_with_sift_box(
        self,
        book_roi_bgr,
    ) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        Matching + omografia SIFT.
        Ritorna:
            - book_id (o None)
            - punteggio
            - quadrilatero raffinato del libro nella ROI (4x2, float32) o None se omografia non affidabile.
        """
        best_id, best_score, best_H = self._match_and_homography(book_roi_bgr)

        if best_id is None or best_H is None:
            return None, 0.0, None

        # prendi la bbox del template e "trasponila" nella ROI tramite H
        tpl = self.templates[best_id]
        x1_t, y1_t, x2_t, y2_t = tpl["bbox"]
        template_corners = np.float32(
            [[x1_t, y1_t],
             [x2_t, y1_t],
             [x2_t, y2_t],
             [x1_t, y2_t]]
        ).reshape(-1, 1, 2)

        roi_corners = cv2.perspectiveTransform(template_corners, best_H)
        roi_corners = roi_corners.reshape(-1, 2)  # shape (4,2)

        return best_id, best_score, roi_corners

    # ---------- core matching + omografia ----------

    def _match_and_homography(self, book_roi_bgr):
        """
        Esegue:
        - matching locale + colore contro tutti i template
        - calcola omografia SIFT per il template migliore

        Ritorna:
            best_id, best_score, best_H (o None se omografia non affidabile)
        """
        if not self.templates:
            raise RuntimeError("Nessun template caricato in BookIdentifier")

        kp, des = self._compute_features(book_roi_bgr)
        if des is None or len(kp) == 0:
            return None, 0.0, None

        roi_hist = self._compute_color_hist(book_roi_bgr)
        bf = cv2.BFMatcher(self.norm_type, crossCheck=False)

        best_id = None
        best_score = 0.0
        best_good_matches = None  # per calcolare H dopo
        best_kp_tpl = None
        best_kp_roi = kp
        best_hist_tpl = None

        # 1) scegli il template migliore in base a match + colore
        for book_id, data in self.templates.items():
            des_t = data["des"]
            hist_t = data["hist"]
            kp_t = data["kp"]

            matches = bf.knnMatch(des_t, des, k=2)
            good = []
            try:
                for m, n in matches:
                    if m.distance < self.ratio_test * n.distance:
                        good.append(m)
            except ValueError:
                continue

            num_good = len(good)
            if num_good < self.min_good_matches:
                continue

            color_sim = cv2.compareHist(hist_t, roi_hist, cv2.HISTCMP_CORREL)
            if color_sim < self.min_color_similarity:
                continue

            score = self.alpha_local * num_good + self.beta_color * float(color_sim)

            if score > best_score:
                best_score = score
                best_id = book_id
                best_good_matches = good
                best_kp_tpl = kp_t
                best_hist_tpl = hist_t

        if best_id is None or best_good_matches is None:
            return None, 0.0, None

        # 2) omografia SIFT fra template migliore e ROI
        if len(best_good_matches) < 4:
            # non abbastanza punti per stimare H
            return best_id, best_score, None

        src_pts = np.float32(
            [best_kp_tpl[m.queryIdx].pt for m in best_good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [best_kp_roi[m.trainIdx].pt for m in best_good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            self.homography_ransac_thresh
        )

        if H is None:
            return best_id, best_score, None

        # opzionale: controlla quanti inlier ha trovato RANSAC
        inliers = mask.ravel().sum() if mask is not None else 0
        if inliers < 10:  # soglia empirica, puoi alzarla
            return best_id, best_score, None

        return best_id, best_score, H
    
if __name__ == "__main__":
    # 1) Segmentazione del ripiano
    # Load all the scene images from "./dataset/scenes/" and segment each shelf
    import os
    scene_folder = "./dataset/scenes/"
    scene_images = [f for f in os.listdir(scene_folder) if f.endswith(".jpg")]
    
    for scene_file in scene_images:
        shelf_img = cv2.imread(os.path.join(scene_folder, scene_file))
        segmenter = LibrarySegmenter(
            smooth_kernel=11,
            peak_prominence=0.15,
            min_book_width_px=10,
            peak_distance_px=15,
            vertical_kernel_height=35,
            debug=True,
        )
        boxes, debug_info = segmenter.segment_shelf(shelf_img)

        # 2) Costruisci database template (esempio)
        # 3) inizializza identificatore
        identifier = BookIdentifier(
            use_sift=True,          # se SIFT non c'è, passa a ORB
            color_hist_bins=(16, 16, 4),
            ratio_test=0.75,
            min_good_matches=15,    # alza se hai falsi positivi
            min_color_similarity=0.6,
            alpha_local=1.0,
            beta_color=0.5,
        )
        # carica alcune immagini template di dorsi già noti
        tpl1 = cv2.imread("./dataset/models/model_0.png")
        tpl2 = cv2.imread("./dataset/models/model_4.png")
        identifier.add_template("book1", tpl1)
        identifier.add_template("book2", tpl2)

        # Carica tutti i template da una cartella
        import os
        template_folder = "./dataset/models/"
        template_images = [f for f in os.listdir(template_folder) if f.endswith(".png")]
        for tpl_file in template_images:
            book_id = os.path.splitext(tpl_file)[0]  # nome file senza estensione
            tpl_img = cv2.imread(os.path.join(template_folder, tpl_file))
            identifier.add_template(book_id, tpl_img)

        # 5) riconosci ogni libro segmentato
        vis = shelf_img.copy()
        for (x1, y1, x2, y2) in boxes:
            roi = shelf_img[y1:y2, x1:x2]

            book_id, score, quad = identifier.match_book_with_sift_box(roi)

            if book_id is not None and quad is not None:
                # quad è in coordinate della ROI: trasla nelle coordinate globali
                quad_global = quad.copy()
                quad_global[:, 0] += x1
                quad_global[:, 1] += y1

                # disegna poligono raffinato SIFT
                pts = quad_global.reshape((-1, 1, 2)).astype(int)
                cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                label = f"{book_id}"
                cv2.putText(
                    vis,
                    label,
                    (int(quad_global[0, 0]), int(quad_global[0, 1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("Recognized Books with SIFT boxes", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()