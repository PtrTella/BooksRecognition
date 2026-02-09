import cv2
import numpy as np
from scipy.signal import find_peaks


class LibrarySegmenter:
    """
    Segmenta i libri su un singolo ripiano usando:
    - gradienti verticali (Sobel)
    - morfologia per enfatizzare i bordi verticali lunghi
    - picchi nel profilo per colonna
    - deskew opzionale tramite Hough
    """

    def __init__(
        self,
        sobel_ksize: int = 3,
        smooth_kernel: int = 21,
        peak_prominence: float = 0.1,
        peak_distance_px: int = 5,
        min_peak_height: float = 0.35,
        min_book_width_px: int = 15,
        vertical_kernel_height: int = 25,
        min_vertical_density: float = 0.15,
        debug: bool = False
    ):
        self.sobel_ksize = sobel_ksize
        self.smooth_kernel = smooth_kernel
        self.peak_prominence = peak_prominence
        self.min_peak_height = min_peak_height
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
            rotated: immagine (eventualmente deskewed)
            (opzionale) debug_dict se debug=True
        """
        # 1) deskew sull'immagine a colori
        rotated, angle_deg = self.auto_rotate_shelves_90_sobel(shelf_bgr)

        # 2) lavora sempre sull'immagine ruotata
        gray_rot = self._to_gray_equalized(rotated)
        edges_vertical = self._vertical_edge_mask(gray_rot)
        profile = self._column_profile(edges_vertical)
        profile_smooth, profile_norm = self._smooth_and_normalize(profile)

        # 3) picchi sul profilo normalizzato
        borders_raw, peaks_props = self._detect_profile_peaks(profile_norm)

        # 4) filtri sui bordi:
        borders_filtered = self._filter_borders_by_vertical_density(
            borders_raw, edges_vertical
        )

        refined_borders = self._refine_borders_by_profile(
            borders_filtered, profile_norm
        )

        # 5) box sull'immagine ruotata
        h, w = gray_rot.shape[:2]
        boxes = self._borders_to_boxes(refined_borders, w, h)
        cutlines = self.compute_oblique_lines(refined_borders, edges_vertical)

        if not self.debug:
            return boxes, rotated

        debug_dict = {
            "gray_rot": gray_rot,
            "rotated": rotated,
            "edges_vertical": edges_vertical,
            "profile": profile,
            "profile_smooth": profile_smooth,
            "profile_norm": profile_norm,
            "borders_raw": borders_raw,
            "borders_filtered": borders_filtered,
            "borders_refined": refined_borders,
            "peaks_properties": peaks_props,
            "deskew_angle_deg": angle_deg,
        }
        return cutlines, rotated, debug_dict

    # ------------ Step interni ------------

    def auto_rotate_shelves_90_sobel(self, img_bgr, sobel_ksize: int = 3, ratio_thresh: float = 1.3):
        """
        Decide se ruotare l'immagine di 90° in base ai gradienti Sobel.

        Assunzioni:
        - le immagini sono quadrate (o quasi).
        - i libri 'in piedi' producono più bordi verticali (Sobel X),
            i libri 'sdraiati' più bordi orizzontali (Sobel Y).

        Parametri:
            img_bgr     : immagine BGR originale
            sobel_ksize : kernel per Sobel (3 è ok)
            ratio_thresh: soglia del rapporto energia_orientata / energia_opposta
                        sopra cui consideriamo l'orientamento 'dominante'

        Ritorna:
            rotated_bgr : immagine (ruotata o identica)
            angle_deg   : 0 o 90 (rotazione applicata, in gradi, clockwise)
        """
        img = img_bgr.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)

        energy_x = float(np.mean(np.abs(gx))) + 1e-6  # bordi verticali (libri in piedi)
        energy_y = float(np.mean(np.abs(gy))) + 1e-6  # bordi orizzontali (libri sdraiati)

        ratio_vert = energy_x / energy_y
        ratio_horiz = energy_y / energy_x

        # Caso 1: più verticali -> libri già in piedi
        if ratio_vert > ratio_thresh:
            return img, 0

        # Caso 2: più orizzontali -> libri sdraiati -> ruota sempre di 90° clockwise
        if ratio_horiz > ratio_thresh:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return rotated, 90

        # Caso ambiguo: lascia stare
        return img, 0
    
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
            height=self.min_peak_height,
        )
        return peaks.tolist(), props

    def _filter_borders_by_vertical_density(self, borders, edges_vertical):
        """
        1. Filtra bordi con densità verticale troppo bassa (scritte / rumore).
        2. Merge dei bordi troppo vicini scegliendo quello con densità maggiore.
        """
        h, w = edges_vertical.shape[:2]
        mid_top = int(0.2 * h)
        mid_bot = int(0.8 * h)

        # Calcola densità verticale per ogni bordo
        candidates = []
        for x in sorted(borders):
            if x < 0 or x >= w:
                continue
            col = edges_vertical[mid_top:mid_bot, x]
            density = float(col.mean()) / 255.0  # 0..1
            if density >= self.min_vertical_density:
                candidates.append((x, density))

        if not candidates:
            return []

        # Merge di bordi troppo vicini usando la densità come criterio
        merged = []
        current_x, current_d = candidates[0]

        for x, d in candidates[1:]:
            if x - current_x < self.min_book_width_px:
                # troppo vicino: tieni quello con densità maggiore
                if d > current_d:
                    current_x, current_d = x, d
            else:
                merged.append(current_x)
                current_x, current_d = x, d

        merged.append(current_x)
        return merged

    def _refine_borders_by_profile(self, borders, profile_norm, window_half_width=5):
        """
        Raffina ogni bordo:
        - cerca il massimo di profile_norm in una finestra ±window_half_width
        - se più bordi finiscono sulla stessa colonna, tieni quello con valore di profilo maggiore.
        """
        w = len(profile_norm)
        refined_candidates = []  # (x_refined, value)

        for x in sorted(borders):
            x0 = max(0, x - window_half_width)
            x1 = min(w - 1, x + window_half_width)
            local = profile_norm[x0:x1 + 1]
            if local.size == 0:
                continue
            local_best_off = int(np.argmax(local))
            x_ref = x0 + local_best_off
            val = float(profile_norm[x_ref])
            refined_candidates.append((x_ref, val))

        # deduplica: se più bordi cadono sulla stessa x_ref, tieni quello col valore più alto
        best_per_x = {}
        for x_ref, val in refined_candidates:
            if x_ref not in best_per_x or val > best_per_x[x_ref]:
                best_per_x[x_ref] = val

        refined = sorted(best_per_x.keys())
        return refined

    def _borders_to_boxes(self, borders, width, height):
        ## INUITLE SE TENIAMO QUELLE OBLIQUE ##
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
    
    def compute_oblique_lines(
        self,
        borders,
        edges_vertical,
        search_radius: int = 5,
        min_points: int = 30,
    ):
        """
        Converte le colonne 'borders' in linee rette oblique che approssimano
        i bordi reali dei libri (derivata di Sobel) in edges_vertical.

        Per ogni bordo:
          1. raccoglie i punti di bordo (x,y) in una fascia orizzontale intorno a x0;
          2. fa una regressione lineare y -> x (x = a*y + b);
          3. genera una retta come due punti (x_top, 0) e (x_bottom, h-1).

        Parametri:
            borders       : lista di x (colonne) candidate ai bordi libro
            edges_vertical: immagine binaria dei bordi verticali (uint8 0/255)
            search_radius : raggio orizzontale (+/-) intorno a ogni x0 per raccogliere punti
            min_points    : minimo numero di punti di bordo richiesti per stimare una retta

        Ritorna:
            lines : lista di linee; ogni linea è ((x1, y1), (x2, y2)), INT
        """
        h, w = edges_vertical.shape[:2]
        lines = []

        if not borders:
            return lines

        borders = sorted(borders)

        # punti di bordo (ys, xs) globali
        ys_all, xs_all = np.where(edges_vertical > 0)

        for x0 in borders:
            # seleziona punti di bordo in una fascia orizzontale [x0 - r, x0 + r]
            x_min = max(0, x0 - search_radius)
            x_max = min(w - 1, x0 + search_radius)

            mask = (xs_all >= x_min) & (xs_all <= x_max)
            xs = xs_all[mask]
            ys = ys_all[mask]

            if xs.size < min_points:
                # troppo pochi punti per stimare una retta: usa linea verticale pura
                x1 = x2 = int(x0)
                y1, y2 = 0, h - 1
                lines.append(((x1, y1), (x2, y2)))
                continue

            # regressione lineare: x = a*y + b
            # (y è indipendente, usiamo np.polyfit sul grado 1)
            a, b = np.polyfit(ys.astype(np.float32), xs.astype(np.float32), 1)

            # calcola intersezioni top e bottom
            y_top = 0
            y_bottom = h - 1
            x_top = a * y_top + b
            x_bottom = a * y_bottom + b

            # clamp ai bordi immagine
            x_top = int(np.clip(round(x_top), 0, w - 1))
            x_bottom = int(np.clip(round(x_bottom), 0, w - 1))

            lines.append(((x_top, y_top), (x_bottom, y_bottom)))

        return lines

from matplotlib import pyplot as plt
segmenter = LibrarySegmenter(debug=True)
import os
#read all shelf images from dataset/scenes/scene_*
shelf_images = []
for file in os.listdir("./dataset/scenes"):
    if file.startswith("scene_"):
        shelf_images.append(os.path.join("./dataset/scenes", file))
        
for shelf_path in shelf_images:
    shelf_bgr = cv2.imread(shelf_path)
    lines, rotated_img, debug = segmenter.segment_shelf(shelf_bgr)

    # Visualizza risultati
    shelf_vis = rotated_img.copy()
    # 1) disegna le cutline oblique (in verde)
    for (p1, p2) in lines:
        cv2.line(
            shelf_vis,
            p1,
            p2,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA
        )


    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(shelf_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original Shelf")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(debug["edges_vertical"], cmap="gray")
    plt.title("Vertical Edges")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.plot(debug["profile_norm"], color="blue")
    plt.scatter(debug["borders_refined"], 
                debug["profile_norm"][debug["borders_refined"]], 
                color="red")
    plt.title("Column Profile with Detected Borders")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(shelf_vis, cv2.COLOR_BGR2RGB))
    plt.title("Detected Book Boxes")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

class BookRecognizer:
    """
    Usa SIFTFeatureExtractor + SIFTMatcher per:
    - memorizzare modelli (template) dei libri
    - riconoscere i libri nelle ROI segmentate
    - restituire una bounding box raffinata via omografia
    """
    def __init__(
        self,
        preprocessor: ImagePreprocessor = ImagePreprocessor(),
        segmenter: LibrarySegmenter = LibrarySegmenter(),
        extractor: SIFTFeatureExtractor = SIFTFeatureExtractor(),
        matcher: SIFTMatcher = SIFTMatcher(),
        ransac_thresh: float = 8.0,
        min_inliers: int = 20,
        min_roi_area: int = 0,
        min_bbox_area: int = 500,
    ):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.matcher = matcher
        self.segmenter = segmenter
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.min_roi_area = min_roi_area
        self.min_bbox_area = min_bbox_area

        # book_id -> dict: img, gray, kp, des, shape
        self.templates = {}

    def add_template(self, book_id: str, img_bgr: np.ndarray):
        """Registra un template (dorso libro) nel database."""
        gray = self.preprocessor.apply_filter(img_bgr)
        kp, des = self.extractor.extract(gray)
        if des is None or len(kp) == 0:
            print(f"[WARN] Nessuna feature per template {book_id}, lo salto.")
            return

        self.templates[book_id] = {
            "img": img_bgr,
            "gray": gray,
            "kp": kp,
            "des": des,
            "shape": gray.shape,
        }

    def recognize_in_roi(self, roi_bgr: np.ndarray):
        """
        Prova a riconoscere un libro in una ROI segmentata.
        Ritorna:
            best_id, best_inliers, best_bbox (in coordinate ROI) oppure (None, 0, None)
        """
        if not self.templates:
            raise RuntimeError("Nessun template registrato nel BookRecognizer")

        gray_roi = self.preprocessor.apply_filter(roi_bgr)
        kp_roi, des_roi = self.extractor.extract(gray_roi)
        if des_roi is None or len(kp_roi) == 0:
            return None, 0, None

        best_id = None
        best_inliers = 0
        best_bbox = None

        for book_id, t in self.templates.items():
            kp_model = t["kp"]
            des_model = t["des"]
            shape_model = t["shape"]

            # 1) matching SIFT con BFMatcher
            good_matches = self.matcher.match(des_model, des_roi)
            if len(good_matches) < self.matcher.min_matches:
                continue

            # 2) omografia
            M, mask, inliers = self.matcher.compute_homography(
                good_matches, kp_model, kp_roi,
                ransac_thresh=self.ransac_thresh,
                min_inliers=self.min_inliers,
            )
            if M is None:
                continue

            # 3) bbox trasformata (nel sistema di coordinate della ROI)
            bbox = self.matcher.transform_bbox(M, shape_model)

            if inliers > best_inliers:
                best_inliers = inliers
                best_id = book_id
                best_bbox = bbox

        if best_id is None:
            return None, 0, None

        return best_id, best_inliers, best_bbox
    
    def recognize_in_scene(self, scene_bgr):
        """
        Riconosce libri in un'immagine scena intera.
        Restituisce lista di risultati:
            [(book_id, inliers, bbox), ...]
        """
        
        # Segmenta ripiani
        lines, rotated_img = self.segmenter.segment_shelf(scene_bgr)
        
        results = []

        for (x1, y1, x2, y2) in lines:
            x1, x2 = sorted((max(0, int(x1)), min(rotated_img.shape[1], int(x2))))
            y1, y2 = sorted((max(0, int(y1)), min(rotated_img.shape[0], int(y2))))

            if x2 - x1 < self.segmenter.min_book_width_px or y2 - y1 <= 0:
                continue

            roi_width = x2 - x1
            roi_height = y2 - y1
            if roi_width * roi_height < self.min_roi_area:
                continue

            roi = rotated_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            book_id, inliers, bbox = self.recognize_in_roi(roi)

            if book_id is not None and bbox is not None:
                bbox_img = bbox + np.array([[x1, y1]])
                bbox_area = cv2.contourArea(bbox_img.reshape(-1, 2).astype(np.float32))
                if bbox_area < self.min_bbox_area:
                    continue
                results.append((book_id, inliers, bbox_img))
        
        return results


# 2) setup riconoscitore (BFMatcher sotto il cofano)
recognizer = BookRecognizer()

# 3) carica tutti i template dalla cartella models
models_dir = "./dataset/models/"
scenes_dir = "./dataset/scenes/"

model_names = [
    f for f in os.listdir(models_dir)
    if os.path.isfile(os.path.join(models_dir, f))
]

for model_name in model_names:
    model_path = os.path.join(models_dir, model_name)
    try:
        tpl_img = cv2.imread(model_path)
        if tpl_img is None:
            print(f"[WARN] Impossibile leggere modello {model_path}, salto.")
            continue
    except Exception as e:
        print(f"[ERROR] Errore caricando modello {model_path}: {e}")
        continue
    
    book_id = os.path.splitext(model_name)[0]  # nome file senza estensione
    recognizer.add_template(book_id, tpl_img)
    print(f"[INFO] Template aggiunto: {book_id}")

if not recognizer.templates:
    print("[ERROR] Nessun template caricato, interrompo.")

# 4) per ogni scena: segmenta e riconosci
scene_names = [
    f for f in os.listdir(scenes_dir)
    if os.path.isfile(os.path.join(scenes_dir, f))
]

for scene_name in scene_names:
    scene_path = os.path.join(scenes_dir, scene_name)
    shelf_img = preprocessor.load_image(scene_path)
    #shelf_img, _, _ = preprocessor.preprocess(scene_path)
    if shelf_img is None:
        print(f"[WARN] Impossibile leggere scena {scene_path}, salto.")
        continue

    print(f"\n[INFO] Elaboro scena: {scene_name}")

    # Riconoscimento libri
    results = recognizer.recognize_in_scene(shelf_img)
    print(f"[INFO] Libri riconosciuti: {len(results)}")
    # Visualizza risultati
    vis = shelf_img.copy()
    for book_id, inliers, bbox in results:
        # Disegna bounding box (assicurati che sia integer array)
        try:
            bbox_int = np.array(bbox, dtype=np.int32)
            cv2.polylines(vis, [bbox_int], isClosed=True, color=(0, 255, 0), thickness=3)
        except Exception:
            print(f"[WARN] Impossibile disegnare bbox per {book_id}")
            # fallback drawing if bbox malformed
            pass

        # Etichetta - prendi il primo punto in maniera robusta
        try:
            pts = np.array(bbox).reshape(-1, 2)
            x, y = int(pts[0, 0]), int(pts[0, 1]) - 10
        except Exception:
            # default position if something goes wrong
            x, y = 10, 30

        cv2.putText(vis, f"{book_id} ({inliers})", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Riconoscimento libri in {scene_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
