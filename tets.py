import cv2
import numpy as np
from scipy.signal import find_peaks


class LibrarySegmenter:
    """
    Segmentazione dei libri su un singolo ripiano usando:
    - gradienti verticali (Sobel)
    - morfologia per enfatizzare bordi verticali lunghi
    - ricerca di picchi nel profilo per colonna
    """

    def __init__(self,
                 sobel_ksize: int = 3,
                 smooth_kernel: int = 31,
                 peak_prominence: float = 0.2,
                 min_book_width_px: int = 25,
                 peak_distance_px: int = 25,
                 vertical_kernel_height: int = 15,
                 debug: bool = False):
        """
        Parametri:
            sobel_ksize          : dimensione kernel Sobel (1,3,5,7)
            smooth_kernel        : size per smoothing del profilo (dispari)
            peak_prominence      : prominenza minima (0..1) per find_peaks sul profilo normalizzato
            min_book_width_px    : larghezza minima plausibile di un libro
            peak_distance_px     : distanza minima fra picchi nel profilo
            vertical_kernel_height: altezza kernel morfologico per tenere solo bordi verticali lunghi
            debug                : se True, ritorna info intermedie
        """
        self.sobel_ksize = sobel_ksize
        self.smooth_kernel = smooth_kernel
        self.peak_prominence = peak_prominence
        self.min_book_width_px = min_book_width_px
        self.peak_distance_px = peak_distance_px
        self.vertical_kernel_height = vertical_kernel_height
        self.debug = debug

    # ----------------- API principale -----------------

    def segment_shelf(self, shelf_bgr):
        """
        Esegue l'intera pipeline su un'immagine BGR di un singolo ripiano.

        Ritorna:
            boxes: lista di [x1, y1, x2, y2]
            debug_dict (se self.debug=True) con info intermedie.
        """
        gray = self._preprocess_gray(shelf_bgr)
        abs_grad_x = self._compute_vertical_gradient(gray)
        edges_vertical = self._keep_strong_vertical_edges(abs_grad_x)

        profile = self._build_column_profile(edges_vertical)
        profile_smooth, profile_norm = self._smooth_and_normalize_profile(profile)
        borders_raw, peaks_props = self._find_borders_from_profile(profile_smooth)

        # opzionale: raffinamento locale di ogni bordo usando la mappa di gradienti
        refined_borders = self._refine_borders_locally(borders_raw, abs_grad_x)

        # costruisci bounding box da bordi raffinati
        h, w = gray.shape[:2]
        boxes = self._borders_to_boxes(refined_borders, w, h)

        debug_dict = None
        if self.debug:
            debug_dict = {
                "gray": gray,
                "abs_grad_x": abs_grad_x,
                "edges_vertical": edges_vertical,
                "profile": profile,
                "profile_smooth": profile_smooth,
                "profile_norm": profile_norm,
                "borders_raw": borders_raw,
                "borders_refined": refined_borders,
                "peaks_properties": peaks_props,
            }

        return (boxes, debug_dict) if self.debug else boxes

    # ----------------- Step di pipeline -----------------

    def _preprocess_gray(self, bgr_img):
        """Converti in grigio + equalizzazione dell'istogramma."""
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        return gray_eq

    def _compute_vertical_gradient(self, gray_img):
        """Calcola gradiente orizzontale (bordi verticali) con Sobel."""
        grad_x = cv2.Sobel(
            gray_img,
            cv2.CV_64F,
            1,  # dx
            0,  # dy
            ksize=self.sobel_ksize
        )
        abs_grad_x = np.abs(grad_x)
        return abs_grad_x

    def _keep_strong_vertical_edges(self, abs_grad_x):
        """
        Tieni solo bordi verticali forti e sufficientemente lunghi:
        - threshold (Otsu) sulla mappa dei gradienti
        - apertura morfologica con kernel verticale
        """
        # da float64 a uint8 per threshold
        grad_uint8 = cv2.convertScaleAbs(abs_grad_x)

        # threshold automatico (Otsu)
        _, edges_strong = cv2.threshold(
            grad_uint8, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # kernel verticale per enfatizzare strutture verticali lunghe
        kernel_vertical = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, self.vertical_kernel_height)
        )
        edges_vertical = cv2.morphologyEx(
            edges_strong,
            cv2.MORPH_OPEN,
            kernel_vertical
        )
        return edges_vertical

    def _build_column_profile(self, edges_vertical):
        """
        Costruisce il profilo per colonna sommando i pixel dei bordi verticali.
        """
        # somma su tutte le righe (axis=0)
        col_profile = edges_vertical.sum(axis=0).astype(np.float64)
        return col_profile

    def _smooth_and_normalize_profile(self, profile):
        """Applica smoothing e normalizzazione (0..1) al profilo."""
        if self.smooth_kernel > 1:
            k = self.smooth_kernel
            kernel = np.ones(k, dtype=np.float64) / float(k)
            profile_smooth = np.convolve(profile, kernel, mode='same')
        else:
            profile_smooth = profile.copy()

        if profile_smooth.max() > 0:
            profile_norm = profile_smooth / profile_smooth.max()
        else:
            profile_norm = profile_smooth

        return profile_smooth, profile_norm

    def _find_borders_from_profile(self, profile_smooth):
        """
        Trova i bordi dei libri come picchi nel profilo smoothato.
        Usa prominenza e distanza minima per rimuovere picchi spuri.
        """
        # normalizza per usare parametri in [0..1]
        if profile_smooth.max() > 0:
            profile_norm = profile_smooth / profile_smooth.max()
        else:
            profile_norm = profile_smooth

        prominence_abs = self.peak_prominence  # è già una frazione del max
        distance_px = self.peak_distance_px

        peaks, properties = find_peaks(
            profile_norm,
            prominence=prominence_abs,
            distance=distance_px
        )

        # Aggiungi bordi estremi 0 e w dopo, in _borders_to_boxes
        borders = peaks.tolist()
        return borders, properties

    def _refine_borders_locally(self, borders, abs_grad_x, window_half_width=3):
        """
        Per ogni bordo candidato x, cerca localmente (±window_half_width)
        la colonna con gradiente medio più alto e usa quella posizione.
        """
        h, w = abs_grad_x.shape[:2]
        refined = []

        for x in sorted(borders):
            x0 = max(1, x - window_half_width)
            x1 = min(w - 2, x + window_half_width)

            local_slice = abs_grad_x[:, x0:x1 + 1]
            col_strength = local_slice.mean(axis=0)
            best_offset = int(np.argmax(col_strength))
            x_refined = x0 + best_offset
            refined.append(x_refined)

        # rimuovi duplicati o bordi troppo vicini rispetto a min_book_width_px
        merged = []
        for x in sorted(refined):
            if not merged:
                merged.append(x)
            else:
                if x - merged[-1] < self.min_book_width_px:
                    # tieni quello con posizione media (opzionale puoi usare forza maggiore)
                    merged[-1] = int((merged[-1] + x) / 2)
                else:
                    merged.append(x)
        return merged

    def _borders_to_boxes(self, borders, img_width, img_height):
        """
        Converte una lista di bordi x in bounding box [x1,y1,x2,y2].
        Aggiunge 0 e img_width come bordi estremi, ripulendo segmenti troppo stretti.
        """
        # aggiungi estremi, poi ordina
        borders_full = [0] + sorted(borders) + [img_width]

        boxes = []
        for i in range(len(borders_full) - 1):
            x1 = borders_full[i]
            x2 = borders_full[i + 1]
            if x2 - x1 < self.min_book_width_px:
                continue
            boxes.append([x1, 0, x2, img_height])
        return boxes


if __name__ == "__main__":
    # Esempio di utilizzo:
    img = cv2.imread("./dataset/scenes/scene_0.jpg")

    segmenter = LibrarySegmenter(
        sobel_ksize=3,
        smooth_kernel=31,
        peak_prominence=0.25,
        min_book_width_px=10,
        peak_distance_px=15,
        vertical_kernel_height=15,
        debug=True
    )

    boxes, debug_info = segmenter.segment_shelf(img)

    vis = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    #cv2.imshow("Detected Book Segments", vis)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

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
