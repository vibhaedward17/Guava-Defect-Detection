"""
Guava Defect Detection & Morphological Analysis Pipeline (v3)
=============================================================
Stage 1: Fruit segmentation (colour + GrabCut refinement).
Stage 2: Defect detection with anatomy-first logic (stem / calyx are identified
         and removed BEFORE lesions are judged).
Stage 3: Colour analysis, shape scores, grading and report.

What is new in v3 (all public names and signatures are unchanged, so
app_gradio.py / app_streamlit.py keep working)
- Illumination is estimated twice (the second time WITHOUT lesion pixels), so big
  lesions no longer drag the "local skin brightness" down and hide themselves.
- Anatomy first: stem and calyx are chosen from the geometry (touching the
  silhouette, small, compact, dark core) and only ONE stem + ONE calyx are allowed.
  Everything else is judged as a possible defect.
- A calyx with rot spreading from it: only the calyx core + a small halo is
  ignored, the spreading rot is still reported as a defect.
- Ring-shaped lesions (dark centre + brown ring) are hole-filled into ONE object.
- Rim shading (thin dark arcs on the silhouette) is rejected.
- Confidence is computed from real features (darkness, contrast, solidity, size)
  instead of fixed numbers.
- Every region keeps its exact pixel mask. Surface-damage % and the colour chart
  are computed from those masks, so boxes, grade and chart always agree.
- Lesion outlines are drawn on the annotated image, not only boxes.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------------------
# Defect & Anatomical Knowledge Base
# ---------------------------------------------------------------------------
DEFECT_CATALOG: Dict[str, Dict[str, str]] = {
    "Anthracnose (Colletotrichum gloeosporioides)": {
        "short_name": "Anthracnose Lesion",
        "appearance": (
            "Dark brown to sunken black circular-to-oval necrotic lesion with a "
            "sharply defined margin and depressed crater-like center. Under high humidity, "
            "concentric rings of salmon-pink spore masses may appear inside the lesion."
        ),
        "cause": "Fungal infection by Colletotrichum gloeosporioides penetrating cuticle micro-fissures.",
        "severity_impact": "High — expands rapidly during post-harvest storage and causes soft tissue decay.",
        "color_hex": "#EF4444",
    },
    "Fruit Canker / Scab (Pestalotiopsis psidii)": {
        "short_name": "Fruit Canker / Scab",
        "appearance": (
            "Raised, corky, rusty-brown to grey-brown scabby eruptions with cracked "
            "epidermal margins. Unlike anthracnose, canker lesions are dry, rough, "
            "and slightly elevated rather than sunken."
        ),
        "cause": "Fungal pathogen Pestalotiopsis psidii infecting developing epicarp tissue.",
        "severity_impact": "Moderate — degrades visual grade and stunts local pericarp expansion.",
        "color_hex": "#F97316",
    },
    "Stylar-End / Phomopsis Rot (Phomopsis psidii)": {
        "short_name": "Stylar / Soft Rot",
        "appearance": (
            "Water-soaked, soft brownish discoloration spreading concentrically "
            "with a diffuse yellowish halo and pulpy underlying flesh."
        ),
        "cause": "Post-harvest fungal invasion through epidermal wounds or physiological breakdown.",
        "severity_impact": "Critical — renders fruit unmarketable and contaminates adjacent batch.",
        "color_hex": "#DC2626",
    },
    "Fruit Fly Oviposition Puncture (Bactrocera correcta)": {
        "short_name": "Insect / Fly Puncture",
        "appearance": (
            "Minute dark pin-hole puncture surrounded by a translucent yellow or "
            "water-soaked halo and slight resinous exudate."
        ),
        "cause": "Oviposition sting by guava fruit fly (Bactrocera spp.).",
        "severity_impact": "High — internal larval tunneling causes rapid core liquefaction.",
        "color_hex": "#EAB308",
    },
}

LABEL_TO_CATALOG = {
    "Anthracnose Lesion": "Anthracnose (Colletotrichum gloeosporioides)",
    "Fruit Canker / Scab": "Fruit Canker / Scab (Pestalotiopsis psidii)",
    "Stylar / Soft Rot": "Stylar-End / Phomopsis Rot (Phomopsis psidii)",
    "Insect / Fly Puncture": "Fruit Fly Oviposition Puncture (Bactrocera correcta)",
}

SUPPRESSED_ANATOMY_INFO: Dict[str, str] = {
    "Stem / Pedicel (Ignored)": "Small woody stalk at the apical pole.",
    "Blossom-End Calyx (Ignored)": "Small dry calyx crown remnant.",
    "Natural Lenticels / Russeting (Ignored)": "Harmless micro-pores / superficial russeting.",
}


@dataclass
class DetectionRegion:
    label: str
    category: str  # "DEFECT" or "SUPPRESSED_ANATOMY"
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_pct: float
    suppression_reason: str = ""
    pathology_key: Optional[str] = None
    # v3: exact pixels of the region (full-image boolean mask). Optional, so old code still works.
    pixel_mask: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class PipelineOutput:
    original_rgb: np.ndarray
    yolov8_mask: np.ndarray
    segmented_guava_rgb: np.ndarray
    annotated_yolov5_rgb: np.ndarray
    defects: List[DetectionRegion] = field(default_factory=list)
    suppressed_regions: List[DetectionRegion] = field(default_factory=list)
    color_distribution: Dict[str, float] = field(default_factory=dict)
    shape_metrics: Dict[str, float] = field(default_factory=dict)
    shape_distribution: Dict[str, float] = field(default_factory=dict)
    shape_classification: str = "Globose / Classic Round Spheroid"
    health_status: str = "Grade A"
    diagnostic_markdown: str = ""
    chart_figure: Optional[plt.Figure] = None
    surface_damage_pct: float = 0.0
    grade: str = "A"


def _grade_from_damage(damage_pct: float) -> str:
    if damage_pct < 1.0:
        return "A"
    if damage_pct < 5.0:
        return "B"
    if damage_pct < 15.0:
        return "C"
    return "Reject"


GRADE_TEXT = {
    "A": "Healthy / premium",
    "B": "Minor blemishes",
    "C": "Noticeable damage",
    "Reject": "Unmarketable",
}


def _bbox_of(mask_bool: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_bool)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class GuavaDefectPipeline:
    def __init__(
        self,
        yolov8_seg_weights: str = "yolov8n-seg.pt",
        yolov5_defect_weights: str = "guava_yolov5_defect.pt",
        conf_threshold: float = 0.45,
        polar_exclusion_ratio: float = 0.22,
        max_lenticel_area_pct: float = 0.30,
    ):
        self.conf_threshold = conf_threshold
        self.polar_exclusion_ratio = polar_exclusion_ratio
        self.max_lenticel_area_pct = max_lenticel_area_pct

        # Lesion sensitivity. More negative = stricter, less negative = more sensitive.
        self.brown_dl = -14.0     # brown pixel must be this much darker than local skin
        self.dark_dl = -28.0      # any pixel this much darker than local skin = dark lesion
        self.core_dl = -45.0      # very dark core (calyx / deep lesion centre)
        self.chroma_ratio = 0.30  # 'off-colour' if (b*-a*) < this x the fruit's skin value

        # Anatomy limits (fractions of the fruit's smaller side / of the fruit area)
        self.stem_max_area_pct = 2.5
        self.calyx_max_area_pct = 4.0
        self.calyx_crown_max_pct = 5.0

        self.yolov8_model = None
        self.yolov5_model = None
        self._try_load_models(yolov8_seg_weights, yolov5_defect_weights)

    def _try_load_models(self, yolov8_weights: str, yolov5_weights: str) -> None:
        try:
            from ultralytics import YOLO
            if os.path.exists(yolov8_weights):
                self.yolov8_model = YOLO(yolov8_weights)
        except Exception:
            self.yolov8_model = None

        try:
            import torch
            if os.path.exists(yolov5_weights):
                self.yolov5_model = torch.hub.load(
                    "ultralytics/yolov5", "custom", path=yolov5_weights, local_repo=False
                )
                self.yolov5_model.conf = self.conf_threshold
        except Exception:
            self.yolov5_model = None

    # -----------------------------------------------------------------------
    # Shadow-proof lesion maps
    # -----------------------------------------------------------------------
    def _lesion_maps(self, image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        L = lab[:, :, 0]
        m = (mask > 0).astype(np.float32)
        k = int(min(mask.shape) * 0.25) | 1

        def illum_of(weight: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(L * weight, (k, k), 0) / np.maximum(cv2.GaussianBlur(weight, (k, k), 0), 1e-3)

        illum = illum_of(m)
        dL = L - illum
        # 2nd pass: estimate the skin brightness WITHOUT the lesions, so big lesions cannot hide themselves
        m2 = m * (dL > -18.0).astype(np.float32)
        if float(m2.sum()) > 0.3 * float(m.sum()) and float(m2.sum()) > 100:
            illum = illum_of(m2)
            dL = L - illum

        H, S = hsv[:, :, 0], hsv[:, :, 1]
        brown = (H <= 20) & (S >= 45) & (dL < self.brown_dl)
        dark = dL < self.dark_dl
        necro_dark = (brown | dark) & (mask > 0)

        chroma = lab[:, :, 2] - lab[:, :, 1]
        k_in = max(5, int(min(mask.shape) * 0.02) | 1)
        inner_c = cv2.erode((mask > 0).astype(np.uint8), np.ones((k_in, k_in), np.uint8)) > 0
        if np.count_nonzero(inner_c) < 50:
            inner_c = mask > 0
        ref = float(np.percentile(chroma[inner_c], 75))
        if ref > 10.0:
            chroma_anom = (chroma < self.chroma_ratio * ref) & inner_c
        else:
            chroma_anom = np.zeros_like(necro_dark)
        chroma_anom = cv2.morphologyEx(chroma_anom.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

        necro = necro_dark | chroma_anom
        core = ((dL < self.core_dl) | (L < 45)) & (mask > 0)
        return necro, core, necro_dark

    # -----------------------------------------------------------------------
    # Split touching blobs with a distance-transform watershed
    # -----------------------------------------------------------------------
    @staticmethod
    def _split_blobs(anom: np.ndarray):
        n, lab, st, _ = cv2.connectedComponentsWithStats(anom, connectivity=8)
        out = np.zeros(anom.shape, dtype=np.int32)
        nxt = 1
        for k in range(1, n):
            comp = (lab == k).astype(np.uint8)
            area = int(st[k, cv2.CC_STAT_AREA])
            pieces = [comp]
            if area >= 400:
                cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cs, key=cv2.contourArea)
                sol = cv2.contourArea(c) / max(1.0, cv2.contourArea(cv2.convexHull(c)))
                if sol < 0.93:
                    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
                    seeds = (dist > 0.45 * dist.max()).astype(np.uint8)
                    ns, ls = cv2.connectedComponents(seeds, connectivity=8)
                    if ns - 1 >= 2:
                        markers = ls.astype(np.int32)
                        markers[comp == 0] = ns
                        gray = (255 - dist / max(1e-3, dist.max()) * 255).astype(np.uint8)
                        ws = cv2.watershed(cv2.merge([gray, gray, gray]), markers)
                        cand = [((ws == i) & (comp > 0)).astype(np.uint8) for i in range(1, ns)]
                        if all(int(cp.sum()) >= 0.15 * area for cp in cand):
                            pieces = cand
            for pc in pieces:
                out[pc > 0] = nxt
                nxt += 1
        stats = np.zeros((nxt, 5), dtype=np.int64)
        for k in range(1, nxt):
            ys_, xs_ = np.where(out == k)
            if len(ys_):
                stats[k] = (xs_.min(), ys_.min(), xs_.max() - xs_.min() + 1, ys_.max() - ys_.min() + 1, len(ys_))
        return out, stats, nxt

    # -----------------------------------------------------------------------
    # STAGE 1: Segmentation & Background Removal
    # -----------------------------------------------------------------------
    def segment_guava_yolov8(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w, _ = image_rgb.shape
        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        sat = np.where(maxc > 0, (maxc - minc) / np.maximum(1.0, maxc) * 255.0, 0.0)

        fruit_tissue = (
            ((g - b > 32.0) & (g > 42.0) & (r <= g * 1.40) & (sat > 62.0))
            | ((sat > 98.0) & ((r - b) > 38.0) & (r > g * 1.15) & (b < 85.0) & (g > 18.0))
        ).astype(np.uint8)

        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(fruit_tissue, connectivity=8)
        if num_l <= 1:
            fallback = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(fallback, (w // 2, h // 2), int(min(h, w) * 0.36), 255, -1)
            return fallback, cv2.bitwise_and(image_rgb, image_rgb, mask=fallback)

        best_l = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fruit_comp = (labels == best_l).astype(np.uint8)

        k_rad = max(11, int(min(h, w) * 0.028) | 1)
        closed = cv2.morphologyEx(
            fruit_comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_rad, k_rad))
        )

        inv = (1 - closed).astype(np.uint8)
        nl_i, lbl_i, st_i, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        mask = closed.copy()
        for k in range(1, nl_i):
            x, y, cw, ch, _ = st_i[k]
            if x > 0 and y > 0 and (x + cw) < w and (y + ch) < h:
                mask[lbl_i == k] = 1

        mask_u8 = (mask * 255).astype(np.uint8)

        def _shape_stats(m_u8):
            cs, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                return 0.0, 0.0, False
            c = max(cs, key=cv2.contourArea)
            ar = cv2.contourArea(c)
            hl = max(1.0, cv2.contourArea(cv2.convexHull(c)))
            border = bool(m_u8[0, :].any() or m_u8[-1, :].any() or m_u8[:, 0].any() or m_u8[:, -1].any())
            return ar, ar / hl, border

        area0, sol0, border0 = _shape_stats(mask_u8)
        if border0 or sol0 < 0.85:
            try:
                sc = min(1.0, 480.0 / max(h, w))
                sw, sh = max(8, int(w * sc)), max(8, int(h * sc))
                small = cv2.resize(image_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
                m_s = cv2.resize(mask_u8, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
                sat_s = cv2.resize(sat.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
                sat_blur = cv2.GaussianBlur(np.clip(sat_s, 0, 255).astype(np.uint8), (5, 5), 0)
                thr, _ = cv2.threshold(sat_blur[m_s].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cand = (m_s & (sat_blur > thr)).astype(np.uint8)
                k_s = max(3, int(min(sh, sw) * 0.03) | 1)
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_s, k_s))
                cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, ker)
                nc, lc, stc, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
                if nc > 1:
                    cand = (lc == 1 + int(np.argmax(stc[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
                cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, ker)
                cnts_c, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(cand)
                cv2.drawContours(filled, cnts_c, -1, 1, -1)

                gc = np.full((sh, sw), cv2.GC_BGD, np.uint8)
                gc[cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_s + 1, 2 * k_s + 1))) > 0] = cv2.GC_PR_BGD
                gc[filled > 0] = cv2.GC_PR_FGD
                gc[cv2.erode(filled, ker) > 0] = cv2.GC_FGD
                if np.count_nonzero(gc == cv2.GC_FGD) > 50:
                    bgd = np.zeros((1, 65), np.float64)
                    fgd = np.zeros((1, 65), np.float64)
                    cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
                    fg_s = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
                    fg = cv2.resize(fg_s, (w, h), interpolation=cv2.INTER_NEAREST)
                    k2 = max(5, int(min(h, w) * 0.02) | 1)
                    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
                    n2, l2, s2, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
                    if n2 > 1:
                        b2 = 1 + int(np.argmax(s2[1:, cv2.CC_STAT_AREA]))
                        fg = (l2 == b2).astype(np.uint8)
                        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_rad, k_rad)))
                        inv2 = (1 - fg).astype(np.uint8)
                        nl2, lb2, st2, _ = cv2.connectedComponentsWithStats(inv2, connectivity=4)
                        for kk in range(1, nl2):
                            bx, by, bw_, bh_, _ = st2[kk]
                            if bx > 0 and by > 0 and (bx + bw_) < w and (by + bh_) < h:
                                fg[lb2 == kk] = 1
                        new_u8 = (fg * 255).astype(np.uint8)
                        area1, sol1, border1 = _shape_stats(new_u8)
                        if area1 >= 0.02 * h * w and (sol1 >= sol0 or (border0 and not border1)):
                            mask_u8 = new_u8
            except Exception:
                pass

        # Fill shallow notches (dark/shadowed skin the colour rule dropped). Accept only a small gain.
        try:
            ys_m, xs_m = np.where(mask_u8 > 0)
            dm = min(xs_m.max() - xs_m.min(), ys_m.max() - ys_m.min())
            kk = max(9, int(dm * 0.22) | 1)
            padm = kk
            m_pad = cv2.copyMakeBorder(mask_u8, padm, padm, padm, padm, cv2.BORDER_CONSTANT, value=0)
            closed2 = cv2.morphologyEx(m_pad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
            closed2 = closed2[padm:-padm, padm:-padm]
            gain = (np.count_nonzero(closed2) - np.count_nonzero(mask_u8)) / max(1, np.count_nonzero(mask_u8))
            if 0 < gain <= 0.08:
                mask_u8 = closed2
        except Exception:
            pass

        mask_u8 = self._fill_notches(mask_u8, image_rgb)
        segmented_rgb = np.where(mask_u8[:, :, None] > 0, image_rgb, 18).astype(np.uint8)
        return mask_u8, segmented_rgb

    @staticmethod
    def _fill_notches(mask_u8: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
        """A very dark lesion can be cut out of the mask (it fails the 'green/yellow tissue' test).
        Fill thick notches between the mask and its convex hull when they do not look like background."""
        cs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return mask_u8
        c = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        if area / max(1.0, cv2.contourArea(hull)) < 0.80:
            return mask_u8  # too ragged, unsafe to fill
        hm = np.zeros_like(mask_u8)
        cv2.drawContours(hm, [hull], -1, 255, -1)
        gap = ((hm > 0) & (mask_u8 == 0)).astype(np.uint8)
        gap = cv2.morphologyEx(gap, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
        _, _, bw, bh = cv2.boundingRect(c)
        dmin = float(min(bw, bh))
        border = np.concatenate([image_rgb[0], image_rgb[-1], image_rgb[:, 0], image_rgb[:, -1]]).astype(np.float32)
        bg = np.median(border, axis=0)
        out = mask_u8.copy()
        for k in range(1, n):
            comp = lbl == k
            a = float(st[k, cv2.CC_STAT_AREA])
            if a < 0.003 * area or a > 0.15 * area:
                continue
            if cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5).max() < 0.05 * dmin:
                continue  # thin sliver (e.g. beside a stalk)
            if np.linalg.norm(image_rgb[comp].astype(np.float32).mean(axis=0) - bg) < 30.0:
                continue  # it really is background
            out[comp] = 255
        return out

    # ------------------------------------------------------------------
    # STAGE 2 helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fill_holes(u8: np.ndarray, max_hole_px: Optional[int] = None) -> np.ndarray:
        """Fill ENCLOSED holes only (a ring lesion with a dark centre becomes ONE object).
        Concavities that open to the outside are never filled."""
        inv = (1 - u8).astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        out = u8.copy()
        H, W = u8.shape
        for k in range(1, n):
            x, y, cw, ch, a = st[k]
            if x > 0 and y > 0 and x + cw < W and y + ch < H:
                if max_hole_px is None or a <= max_hole_px:
                    out[lbl == k] = 1
        return out

    def _components(self, anom_u8: np.ndarray) -> List[np.ndarray]:
        lbl, st, n = self._split_blobs(anom_u8)
        return [(lbl == k) for k in range(1, n) if st[k, 4] > 0]

    @staticmethod
    def _features(comp: np.ndarray, lab: np.ndarray, r: np.ndarray, g: np.ndarray,
                  dist_map: np.ndarray, core_map: np.ndarray, fruit_area: float) -> Dict:
        cu = comp.astype(np.uint8)
        x1, y1, x2, y2 = _bbox_of(comp)
        cw, ch = x2 - x1, y2 - y1
        cs, _ = cv2.findContours(cu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        carea = max(1.0, cv2.contourArea(c))
        hull = max(1.0, cv2.contourArea(cv2.convexHull(c)))
        per = max(1.0, cv2.arcLength(c, True))
        a = int(np.count_nonzero(comp))
        d = dist_map[comp]
        return {
            "mask": comp,
            "area": a,
            "area_pct": a / fruit_area * 100.0,
            "bbox": (x1, y1, x2, y2),
            "cw": cw,
            "ch": ch,
            "aspect": max(cw, ch) / max(1.0, float(min(cw, ch))),
            "solidity": min(1.0, carea / hull),
            "circ": min(1.0, 4.0 * np.pi * carea / (per * per)),
            "mean_l": float(np.mean(lab[:, :, 0][comp])),
            "mean_a": float(np.mean(lab[:, :, 1][comp])),
            "mean_r": float(np.mean(r[comp])),
            "mean_g": float(np.mean(g[comp])),
            "edge_d": float(d.min()),
            "depth": float(d.max()),
            "core_frac": float(np.mean(core_map[comp])),
        }

    # ------------------------------------------------------------------
    # STAGE 2: Defect detection + anatomical suppression
    # ------------------------------------------------------------------
    def detect_defects_yolov5(
        self, image_rgb: np.ndarray, mask: np.ndarray
    ) -> Tuple[List[DetectionRegion], List[DetectionRegion]]:
        h, w = mask.shape
        fruit = mask > 0
        ys, xs = np.where(fruit)
        if len(ys) == 0:
            return [], []

        fy1, fy2, fx1, fx2 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        fh, fw = max(1, fy2 - fy1), max(1, fx2 - fx1)
        dmin = float(min(fw, fh))
        fruit_area = float(np.count_nonzero(fruit))

        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        erode_px = max(3, int(min(h, w) * 0.006) | 1)
        inner_mask = cv2.erode(mask, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)) > 0
        dist_map = cv2.distanceTransform(fruit.astype(np.uint8), cv2.DIST_L2, 5)

        necro_map, core_map, _ = self._lesion_maps(image_rgb, mask)

        healthy_px = inner_mask & (r <= g * 1.20) & (g > 60.0) & (~necro_map)
        has_h = bool(np.any(healthy_px))
        med_r = float(np.median(r[healthy_px])) if has_h else 140.0
        med_g = float(np.median(g[healthy_px])) if has_h else 140.0
        med_a = float(np.median(lab[:, :, 1][healthy_px])) if has_h else 112.0
        med_l = float(np.median(lab[:, :, 0][healthy_px])) if has_h else 150.0
        med_rg = med_r / max(1.0, med_g)
        med_diff = med_r - med_g

        # Combined anomaly map (shadow-proof necrosis + a few colour rules for brown scab)
        anom0 = (
            (inner_mask | (dist_map > 0))  # keep the rim; thin rim artefacts are rejected later
            & (
                ((r > g * (med_rg + 0.22)) & ((r - g) >= max(13.0, med_diff + 12.0)) & (g < med_g * 0.68))
                | ((g < med_g * 0.45) & (r >= g * 1.08) & (g < 58.0))
                | ((g < 36.0) & (r >= g))
                | necro_map
            )
        ).astype(np.uint8)
        anom0 = cv2.morphologyEx(anom0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        close_k = max(3, int(min(h, w) * 0.005) | 1)
        anom0 = cv2.morphologyEx(anom0, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

        min_px = max(14, int(fruit_area * 0.00015))
        suppressed: List[DetectionRegion] = []
        carve = np.zeros((h, w), dtype=np.uint8)
        pad = max(6, int(min(h, w) * 0.012))
        carve_k = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)

        def add_anatomy(label: str, comp: np.ndarray, conf: float, reason: str):
            x1, y1, x2, y2 = _bbox_of(comp)
            suppressed.append(
                DetectionRegion(
                    label=label,
                    category="SUPPRESSED_ANATOMY",
                    confidence=round(float(conf), 2),
                    bbox=(max(0, x1 - pad // 2), max(0, y1 - pad // 2), min(w - 1, x2 + pad // 2), min(h - 1, y2 + pad // 2)),
                    area_pct=round(float(np.count_nonzero(comp)) / fruit_area * 100.0, 2),
                    suppression_reason=reason,
                    pixel_mask=comp.copy(),
                )
            )

        # ---------------- Pass 1: find anatomy ----------------
        comps1 = self._components(self._fill_holes(anom0, int(fruit_area * 0.03)) & (mask > 0).astype(np.uint8))
        feats1 = [
            self._features(c, lab, r, g, dist_map, core_map, fruit_area)
            for c in comps1 if int(np.count_nonzero(c)) >= max(25, int(fruit_area * 0.0008))
        ]

        def attached_to_bigger(f) -> bool:
            """True if this blob touches a clearly bigger damaged blob = it is part of a lesion, not anatomy."""
            dil = cv2.dilate(f["mask"].astype(np.uint8), np.ones((pad * 6 + 1, pad * 6 + 1), np.uint8)) > 0
            for o in feats1:
                if o is not f and o["area"] >= 2.5 * f["area"] and (dil & o["mask"]).any():
                    return True
            return False

        # Stem (a): a thin protrusion that sticks out of the round fruit body (works for green stems too)
        stem = None
        k_o = max(7, int(dmin * 0.18) | 1)
        body = cv2.morphologyEx(fruit.astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_o, k_o)))
        prot = ((fruit) & (body == 0)).astype(np.uint8)
        prot = cv2.morphologyEx(prot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_p, l_p, s_p, _ = cv2.connectedComponentsWithStats(prot, connectivity=8)
        best_p = None
        for k in range(1, n_p):
            x, y, cw, ch, a = s_p[k]
            comp = l_p == k
            ap = a / fruit_area * 100.0
            if 0.05 <= ap <= 3.0 and max(cw, ch) >= 0.09 * dmin and max(cw, ch) / max(1.0, min(cw, ch)) >= 1.8:
                if best_p is None or a > best_p[0]:
                    best_p = (a, comp)
        if best_p is not None:
            comp = best_p[1].copy()
            # a dried, brown base / crown right next to the stalk belongs to the same stem-calyx complex
            grow_s = cv2.dilate(comp.astype(np.uint8), np.ones((pad * 6 + 1, pad * 6 + 1), np.uint8)) > 0
            for o in feats1:
                if (o["edge_d"] <= 0.08 * dmin and o["area_pct"] <= 5.0
                        and (grow_s & o["mask"]).any() and not attached_to_bigger(o)):
                    comp |= o["mask"]
            carve |= cv2.dilate(comp.astype(np.uint8), carve_k)
            add_anatomy("Stem / Pedicel (Ignored)", comp, 0.95, "Stalk sticking out of the fruit body, with its dried base.")
            stem = {"mask": comp, "edge_d": 0.0, "area": int(np.count_nonzero(comp))}

        # Stem (b): small, compact, brownish blob ON the silhouette that is not part of a lesion.
        for f in []:  # brown-blob stem rule disabled: it mislabelled lesion edges
            if attached_to_bigger(f) or f["area_pct"] < 0.3:
                continue
            if (
                f["edge_d"] <= 0.04 * dmin
                and f["area_pct"] <= self.stem_max_area_pct
                and f["cw"] <= 0.32 * fw and f["ch"] <= 0.32 * fh
                and f["aspect"] <= 4.5
                and f["mean_r"] >= f["mean_g"] * 0.95       # woody / brown, not a purple lesion
                and f["depth"] <= 0.10 * dmin               # a stem is thin: it never goes deep into the fruit
            ):
                if stem is None or (f["edge_d"], -f["area"]) < (stem["edge_d"], -stem["area"]):
                    stem = f
        if stem is not None and not any(r_.label.startswith("Stem") for r_ in suppressed):
            comp = stem["mask"]
            carve |= cv2.dilate(comp.astype(np.uint8), carve_k)
            add_anatomy("Stem / Pedicel (Ignored)", comp, 0.90, "Small thin woody structure touching the fruit outline.")

        # Calyx (a): very dark compact core near the edge, with a modest crown around it.
        core_u8 = cv2.morphologyEx(core_map.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nl_c, lbl_c, st_c, _ = cv2.connectedComponentsWithStats(core_u8, connectivity=8)
        core_cands = []
        for k in range(1, nl_c):
            x, y, cw, ch, a = st_c[k]
            comp = lbl_c == k
            if (
                a >= max(50, int(fruit_area * 0.0025))
                and float(np.mean(lab[:, :, 0][comp])) < 42.0
                and a / fruit_area * 100.0 <= self.calyx_max_area_pct
                and cw <= 0.30 * fw and ch <= 0.30 * fh
                and float(dist_map[comp].min()) < 0.14 * dmin
                and (stem is None or not (carve[comp] > 0).any())
            ):
                core_cands.append((float(dist_map[comp].min()), -int(a), k))
        has_calyx = False
        if core_cands:
            core_cands.sort()
            k = core_cands[0][2]
            core_best = (lbl_c == k)
            big_k = np.ones((pad * 4 + 1, pad * 4 + 1), np.uint8)
            nl_n, lbl_n = cv2.connectedComponents(necro_map.astype(np.uint8), connectivity=8)
            touching = np.unique(lbl_n[cv2.dilate(core_best.astype(np.uint8), big_k) > 0])
            grown = core_best.copy()
            for t in touching:
                if t != 0:
                    grown |= (lbl_n == t)
            g_pct = float(np.count_nonzero(grown)) / fruit_area * 100.0
            core_ratio = float(np.count_nonzero(core_best)) / max(1, int(np.count_nonzero(grown)))
            if g_pct <= self.calyx_crown_max_pct and core_ratio <= 0.40:
                carve |= cv2.dilate(grown.astype(np.uint8), carve_k)
                add_anatomy("Blossom-End Calyx (Ignored)", grown, 0.94,
                            "Small very dark compact crown near the fruit edge.")
                has_calyx = True
            # else: a dark centre inside a big damaged area is the lesion itself -> keep it as a defect

        # Calyx (b): dried brown crown without a black core, hugging the outline.
        if not has_calyx:
            best = None
            for f in feats1:
                if stem is not None and f["mask"] is stem["mask"]:
                    continue
                if attached_to_bigger(f):
                    continue
                if (
                    f["edge_d"] <= 0.05 * dmin
                    and 0.5 <= f["area_pct"] <= (12.0 if f["depth"] <= 0.22 * dmin else 6.0)
                    and f["mean_l"] < 110.0
                    and f["cw"] <= 0.55 * fw and f["ch"] <= 0.35 * fh
                    and f["mean_g"] < 0.85 * med_g
                    and f["mean_r"] >= f["mean_g"] * 0.95
                    and f["solidity"] >= 0.70
                    and not (carve[f["mask"]] > 0).all()
                ):
                    if best is None or f["edge_d"] < best["edge_d"]:
                        best = f
            if best is not None:
                crown = best["mask"].copy()
                grow = cv2.dilate(crown.astype(np.uint8), np.ones((pad * 6 + 1, pad * 6 + 1), np.uint8)) > 0
                for o in feats1:   # absorb the rest of the same dried crown (pieces near the outline)
                    if o is not best and o["edge_d"] <= 0.08 * dmin and o["area_pct"] <= 3.0 and (grow & o["mask"]).any():
                        crown |= o["mask"]
                carve |= cv2.dilate(crown.astype(np.uint8), carve_k)
                add_anatomy("Blossom-End Calyx (Ignored)", crown, 0.88, "Dried dark crown touching the fruit outline.")

        # ---------------- Pass 2: lesions (anatomy removed) ----------------
        anom = anom0.copy()
        anom[carve > 0] = 0
        anom = cv2.morphologyEx(anom, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        big_close = max(5, int(min(h, w) * 0.018) | 1)
        anom = cv2.morphologyEx(anom, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big_close, big_close)))
        anom[dist_map < 0.03 * dmin] = 0  # thin outline band: rim shading, not lesion
        # a little wider band: only really DARK pixels count there (pale brown edge shading does not)
        anom[(dist_map < 0.06 * dmin) & (lab[:, :, 0] >= 0.62 * med_l)] = 0
        anom = self._fill_holes(anom, int(fruit_area * 0.03)) & (mask > 0).astype(np.uint8)
        anom[carve > 0] = 0
        comps2 = self._components(anom)

        cand_nat = []
        raw: List[Dict] = []
        for comp in comps2:
            a = int(np.count_nonzero(comp))
            if a < min_px:
                continue
            f = self._features(comp, lab, r, g, dist_map, core_map, fruit_area)
            ap = f["area_pct"]

            # rim shading = thin arc hugging the outline
            if f["depth"] < 0.03 * dmin and f["aspect"] >= 2.5:
                continue
            # shadow-shaped blob
            if ap > 3.0 and f["solidity"] < 0.55:
                continue

            if ap < max(self.max_lenticel_area_pct, 0.95):
                small_dark = ap >= 0.15 and f["mean_l"] < 65.0
                if not small_dark:
                    cand_nat.append((f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3], ap, comp))
                    continue
            raw.append(f)

        # ---------------- Classification ----------------
        defects: List[DetectionRegion] = []
        for f in raw:
            ap = f["area_pct"]
            contrast = max(0.0, (med_l - f["mean_l"]) / max(1.0, med_l))           # 0..1 how much darker than skin
            dark_frac = float(np.mean(lab[:, :, 0][f["mask"]] < 75.0))
            dark = f["mean_g"] < 58.0 or f["mean_l"] < 88.0 or dark_frac > 0.15
            compact = 0.5 * f["solidity"] + 0.5 * f["circ"]

            if dark and (ap >= 0.8 or f["mean_l"] < 45.0):
                label = "Anthracnose Lesion"
            elif dark:
                label = "Insect / Fly Puncture"
            elif ap > 8.0:
                label = "Stylar / Soft Rot"
            elif (f["mean_r"] - f["mean_g"]) >= 16.0 or f["mean_a"] > med_a + 10.0:
                label = "Fruit Canker / Scab"
            else:
                label = "Insect / Fly Puncture" if ap < 0.6 else "Fruit Canker / Scab"

            conf = 0.60 + 0.22 * min(1.0, contrast * 2.5) + 0.12 * compact + 0.06 * min(1.0, ap / 3.0)
            conf = float(min(0.98, max(0.50, conf)))
            conf = min(conf, 0.65 + 0.33 * min(1.0, ap / 2.0))   # tiny spots can never be 90%+ sure
            if conf < self.conf_threshold:
                continue
            defects.append(
                DetectionRegion(
                    label=label,
                    category="DEFECT",
                    confidence=round(conf, 2),
                    bbox=self._pad_box(f["bbox"], w, h),
                    area_pct=round(ap, 2),
                    pathology_key=LABEL_TO_CATALOG.get(label),
                    pixel_mask=f["mask"],
                )
            )

        defects = self._merge_defects(defects, fruit_area, w, h)
        defects.sort(key=lambda d: d.area_pct, reverse=True)

        # one clean, non-overlapping natural skin mark for display
        sep = max(25, int(min(h, w) * 0.04))
        for nx1, ny1, nx2, ny2, nap, ncomp in sorted(cand_nat, key=lambda t: -t[4]):
            clash = False
            for reg in suppressed + defects:
                rx1, ry1, rx2, ry2 = reg.bbox
                if nx1 <= rx2 + sep and nx2 >= rx1 - sep and ny1 <= ry2 + sep and ny2 >= ry1 - sep:
                    clash = True
                    break
            if not clash:
                suppressed.append(
                    DetectionRegion(
                        label="Natural Skin Mark / Lenticel (Ignored)",
                        category="SUPPRESSED_ANATOMY",
                        confidence=0.92,
                        bbox=(nx1, ny1, nx2, ny2),
                        area_pct=round(nap, 3),
                        suppression_reason=f"Area ({nap:.2f}%) below the defect threshold; harmless lenticel/russeting.",
                        pixel_mask=ncomp,
                    )
                )
                break

        return defects, suppressed

    @staticmethod
    def _pad_box(bbox, w, h):
        x1, y1, x2, y2 = bbox
        p = max(4, int(min(h, w) * 0.008))
        return (max(0, x1 - p), max(0, y1 - p), min(w - 1, x2 + p), min(h - 1, y2 + p))

    @staticmethod
    def _merge_defects(defects: List[DetectionRegion], fruit_area: float, w: int, h: int) -> List[DetectionRegion]:
        """Merge boxes that overlap strongly or where one sits inside the other (same lesion in pieces)."""
        defects = list(defects)
        changed = True
        while changed and len(defects) > 1:
            changed = False
            for i in range(len(defects)):
                for j in range(i + 1, len(defects)):
                    A, B = defects[i], defects[j]
                    ax1, ay1, ax2, ay2 = A.bbox
                    bx1, by1, bx2, by2 = B.bbox
                    iw = min(ax2, bx2) - max(ax1, bx1)
                    ih = min(ay2, by2) - max(ay1, by1)
                    if iw <= 0 or ih <= 0:
                        continue
                    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
                    if (iw * ih) / max(1, smaller) > 0.50:
                        big = A if A.area_pct >= B.area_pct else B
                        pm = None
                        if A.pixel_mask is not None and B.pixel_mask is not None:
                            pm = A.pixel_mask | B.pixel_mask
                        area_pct = (float(np.count_nonzero(pm)) / fruit_area * 100.0) if pm is not None \
                            else (A.area_pct + B.area_pct)
                        defects[i] = DetectionRegion(
                            label=big.label,
                            category="DEFECT",
                            confidence=big.confidence,
                            bbox=(min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)),
                            area_pct=round(area_pct, 2),
                            pathology_key=big.pathology_key,
                            pixel_mask=pm,
                        )
                        del defects[j]
                        changed = True
                        break
                if changed:
                    break
        return defects

    # ------------------------------------------------------------------
    # STAGE 3a: Colour & shape analysis
    # ------------------------------------------------------------------
    def analyze_color_and_shape(
        self,
        image_rgb: np.ndarray,
        guava_mask: np.ndarray,
        defects: List[DetectionRegion],
        suppressed: Optional[List[DetectionRegion]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], str]:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        fruit = guava_mask > 0
        total_px = max(1, int(np.count_nonzero(fruit)))
        Hh = hsv[:, :, 0][fruit]

        # Ignored anatomy: exact pixels when available, else its box
        natural_full = np.zeros(guava_mask.shape, dtype=bool)
        for s in (suppressed or []):
            if s.pixel_mask is not None:
                natural_full |= s.pixel_mask
            else:
                x1, y1, x2, y2 = s.bbox
                natural_full[y1: y2 + 1, x1: x2 + 1] = True
        natural_full &= fruit

        # Damage = exact pixels of confirmed defects (so chart, boxes and grade always agree)
        if defects and all(d.pixel_mask is not None for d in defects):
            necro_full = np.zeros(guava_mask.shape, dtype=bool)
            for d in defects:
                necro_full |= d.pixel_mask
        elif defects:
            necro_full = np.zeros(guava_mask.shape, dtype=bool)
            for d in defects:
                x1, y1, x2, y2 = d.bbox
                necro_full[y1: y2 + 1, x1: x2 + 1] = True
            nm, _, _ = self._lesion_maps(image_rgb, guava_mask)
            necro_full &= nm
        else:
            necro_full = np.zeros(guava_mask.shape, dtype=bool)
        necro_full &= fruit & ~natural_full

        nat = natural_full[fruit]
        necrotic = necro_full[fruit]
        rest = (~nat) & (~necrotic)
        deep_green = rest & (Hh >= 38)
        light_green = rest & (Hh >= 31) & (Hh < 38)
        yellow = rest & (Hh < 31)

        counts = {
            "Fresh Chlorophyll Green": int(np.count_nonzero(deep_green)),
            "Light Maturing Green": int(np.count_nonzero(light_green)),
            "Ripe Yellow-Green": int(np.count_nonzero(yellow)),
            "Natural Stem / Calyx / Lenticel": int(np.count_nonzero(nat)),
            "Defected / Necrotic Tissue": int(np.count_nonzero(necrotic)),
        }
        color_dist = {k: round(v / total_px * 100.0, 1) for k, v in counts.items()}
        diff = round(100.0 - sum(color_dist.values()), 1)
        biggest = max(color_dist, key=color_dist.get)
        color_dist[biggest] = round(color_dist[biggest] + diff, 1)
        damage_pct = color_dist["Defected / Necrotic Tissue"]

        contours, _ = cv2.findContours(guava_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = max(1.0, cv2.arcLength(cnt, True))
            circularity = min(1.0, (4.0 * np.pi * area) / (perimeter * perimeter))
            hull_area = max(1.0, cv2.contourArea(cv2.convexHull(cnt)))
            solidity = min(1.0, area / hull_area)
            _, _, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / max(1.0, float(bh))
            sphericity_score = min(1.0, 1.0 - abs(1.0 - aspect_ratio))
            if len(cnt) >= 5:
                (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (min(ma, MA) / max(1.0, max(ma, MA))) ** 2)))
            else:
                eccentricity = 0.25
        else:
            circularity, solidity, aspect_ratio, sphericity_score, eccentricity = 0.88, 0.96, 0.94, 0.94, 0.30

        epidermal_integrity = max(0.0, (100.0 - damage_pct * 1.5) / 100.0)
        radial_symmetry = min(1.0, 1.0 - eccentricity * 0.25)

        shape_metrics = {
            "Circularity Index": round(circularity * 100.0, 1),
            "Convexity / Solidity": round(solidity * 100.0, 1),
            "Equatorial Aspect Balance": round(sphericity_score * 100.0, 1),
            "Radial Symmetry": round(radial_symmetry * 100.0, 1),
            "Epidermal Surface Integrity": round(epidermal_integrity * 100.0, 1),
            "Raw Aspect Ratio (W/H)": round(aspect_ratio, 2),
        }
        score_keys = [
            "Circularity Index", "Convexity / Solidity", "Equatorial Aspect Balance",
            "Radial Symmetry", "Epidermal Surface Integrity",
        ]
        shape_dist = {f"{k} ({shape_metrics[k]:.1f}/100)": shape_metrics[k] for k in score_keys}

        if aspect_ratio < 0.89:
            shape_class = "Pyriform / Ovoid (Pear-Shaped Guava)"
        elif aspect_ratio > 1.08:
            shape_class = "Oblate / Equatorial Spheroid"
        else:
            shape_class = "Globose / Classic Round Spheroid"

        return color_dist, shape_metrics, shape_dist, shape_class

    # ------------------------------------------------------------------
    # STAGE 2 rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_tag(canvas, text, box, color, placed, scale, thick):
        H, W = canvas.shape[:2]
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        pw, ph = tw + 10, th + base + 6
        x1, y1, x2, y2 = box
        x = int(min(max(0, x1), max(0, W - pw)))
        y = y1 - ph if y1 - ph >= 0 else min(y2, H - ph)
        y = int(max(0, y))

        def overlaps(yy):
            rr = (x, yy, x + pw, yy + ph)
            return any(not (rr[2] <= p[0] or rr[0] >= p[2] or rr[3] <= p[1] or rr[1] >= p[3]) for p in placed)

        tries = 0
        while overlaps(y) and tries < 12:
            y = y + ph + 2
            if y + ph > H:
                y = max(0, y1 - ph * (tries + 2))
            tries += 1
        y = int(min(max(0, y), max(0, H - ph)))
        placed.append((x, y, x + pw, y + ph))
        cv2.rectangle(canvas, (x, y), (x + pw, y + ph), color, -1)
        cv2.putText(canvas, text, (x + 5, y + th + 2), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)

    def render_annotated_image(
        self,
        image_rgb: np.ndarray,
        guava_mask: np.ndarray,
        defects: List[DetectionRegion],
        suppressed: List[DetectionRegion],
    ) -> np.ndarray:
        h, w = guava_mask.shape
        dim = (image_rgb.astype(np.float32) * 0.35).astype(np.uint8)
        canvas = np.where(guava_mask[:, :, None] > 0, image_rgb, dim).astype(np.uint8)
        contours, _ = cv2.findContours(guava_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (16, 185, 129), max(2, int(min(h, w) / 300)))

        scale = max(0.45, min(h, w) / 1000.0)
        thick = 1 if scale < 0.8 else 2
        box_t = max(2, int(min(h, w) / 280))
        placed: List[Tuple[int, int, int, int]] = []

        def outline(region: DetectionRegion, color):
            if region.pixel_mask is None:
                return
            cs, _ = cv2.findContours(region.pixel_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cs, -1, color, max(1, box_t - 1))

        for d in defects:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (239, 68, 68), box_t)
            outline(d, (255, 255, 0))
        for sup in suppressed:
            x1, y1, x2, y2 = sup.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (14, 165, 233), max(1, box_t - 1))

        for idx, d in enumerate(defects, 1):
            tag = f"#{idx} {d.label} {int(d.confidence * 100)}%"
            self._draw_tag(canvas, tag, d.bbox, (239, 68, 68), placed, scale, thick)
        for sup in suppressed:
            short = sup.label.replace(" (Ignored)", "").split(" /")[0]
            self._draw_tag(canvas, f"Ignored: {short}", sup.bbox, (14, 165, 233), placed, scale, thick)

        return canvas

    # ------------------------------------------------------------------
    # STAGE 3b: Charts
    # ------------------------------------------------------------------
    def create_pie_charts_figure(
        self,
        color_dist: Dict[str, float],
        shape_dist: Dict[str, float],
        shape_class: str,
        shape_metrics: Optional[Dict[str, float]] = None,
    ) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=130, gridspec_kw={"width_ratios": [1, 1.15]})
        fig.patch.set_facecolor("#0F172A")
        fig.subplots_adjust(left=0.04, right=0.97, top=0.84, bottom=0.24, wspace=0.45)

        names = list(color_dist.keys())
        palette = {
            "Fresh Chlorophyll Green": "#16A34A",
            "Light Maturing Green": "#4ADE80",
            "Ripe Yellow-Green": "#FACC15",
            "Natural Stem / Calyx / Lenticel": "#38BDF8",
            "Defected / Necrotic Tissue": "#EF4444",
        }
        shown = [(n, v) for n, v in color_dist.items() if v > 0.0]
        if not shown:
            shown = [("Fresh Chlorophyll Green", 100.0)]
        _, _, autotexts = ax1.pie(
            [v for _, v in shown],
            colors=[palette[n] for n, _ in shown],
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 4.0 else "",
            startangle=140,
            pctdistance=0.76,
            wedgeprops=dict(width=0.45, edgecolor="#0F172A", linewidth=2),
        )
        for t in autotexts:
            t.set_color("#FFFFFF")
            t.set_fontsize(9)
            t.set_weight("bold")
        ax1.set_title("Surface Colour Analysis (inside fruit mask)", color="#F8FAFC",
                      fontsize=12, fontweight="bold", pad=14)
        ax1.legend(
            [plt.Rectangle((0, 0), 1, 1, color=palette[n]) for n in names],
            [f"{n}: {color_dist[n]:.1f}%" for n in names],
            loc="upper center", bbox_to_anchor=(0.5, -0.04),
            frameon=False, labelcolor="#E2E8F0", fontsize=9, ncol=1,
        )

        metrics = shape_metrics or {}
        keys = [
            "Circularity Index", "Convexity / Solidity", "Equatorial Aspect Balance",
            "Radial Symmetry", "Epidermal Surface Integrity",
        ]
        vals = [metrics.get(k, 0.0) for k in keys]
        bar_colors = ["#22C55E" if v >= 85 else "#F59E0B" if v >= 65 else "#EF4444" for v in vals]
        ypos = np.arange(len(keys))[::-1]
        ax2.set_facecolor("#0F172A")
        ax2.barh(ypos, vals, color=bar_colors, height=0.55)
        ax2.set_xlim(0, 100)
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(keys, color="#E2E8F0", fontsize=10)
        ax2.tick_params(axis="x", colors="#94A3B8", labelsize=9)
        ax2.set_xlabel("Score (0–100, higher is better)", color="#94A3B8", fontsize=9)
        for y, v in zip(ypos, vals):
            ax2.text(min(v + 1.5, 97), y, f"{v:.1f}", va="center", ha="left" if v < 90 else "right",
                     color="#F8FAFC" if v < 90 else "#0F172A", fontsize=10, fontweight="bold")
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax2.spines[sp].set_color("#334155")
        ax2.set_title(f"Shape & Morphology Scores\n{shape_class}", color="#F8FAFC",
                      fontsize=12, fontweight="bold", pad=14)
        return fig

    # ------------------------------------------------------------------
    # STAGE 3c: Report
    # ------------------------------------------------------------------
    def build_diagnostic_report(
        self,
        defects: List[DetectionRegion],
        suppressed: List[DetectionRegion],
        shape_class: str,
        shape_metrics: Dict[str, float],
        surface_damage_pct: float = 0.0,
    ) -> Tuple[str, str]:
        defect_area = round(sum(d.area_pct for d in defects), 2)
        total_damage = max(defect_area, surface_damage_pct)
        grade = _grade_from_damage(total_damage)
        status = f"Grade {grade} — {GRADE_TEXT[grade]}"

        main_defect = "None"
        if defects:
            main_defect = max(defects, key=lambda d: d.area_pct).label

        lines = [
            f"## Verdict: **{status}**",
            "",
            "| Surface damage | Defects found | Main defect | Ignored anatomy |",
            "|:--:|:--:|:--:|:--:|",
            f"| **{total_damage:.1f}%** | **{len(defects)}** | **{main_defect}** | **{len(suppressed)}** |",
            "",
            f"- **Shape:** {shape_class} (circularity `{shape_metrics['Circularity Index']}`, "
            f"solidity `{shape_metrics['Convexity / Solidity']}`, aspect ratio `{shape_metrics['Raw Aspect Ratio (W/H)']}`)",
        ]

        if not defects and surface_damage_pct >= 3.0:
            lines.append(
                f"- ⚠️ **Check manually:** no defect box was drawn, but colour analysis found "
                f"`{surface_damage_pct:.1f}%` discoloured tissue."
            )
        lines.append("")

        if defects:
            lines.append("### Confirmed defects")
            for i, d in enumerate(defects, 1):
                info = DEFECT_CATALOG.get(d.pathology_key or "", {})
                lines.append(
                    f"{i}. **{d.label}** — confidence `{d.confidence * 100:.0f}%`, area `{d.area_pct:.2f}%`\n"
                    f"   - **Looks like:** {info.get('appearance', 'Localized necrotic epicarp discoloration.')}\n"
                    f"   - **Cause:** {info.get('cause', 'Fungal or mechanical epicarp injury.')}\n"
                    f"   - **Impact:** {info.get('severity_impact', 'Requires sorting prior to packing.')}"
                )
        else:
            lines.append("### Confirmed defects: none")

        if suppressed:
            lines.append("\n### Ignored natural features")
            for s in suppressed:
                lines.append(f"- **{s.label.replace(' (Ignored)', '')}** (`{s.area_pct:.2f}%`): {s.suppression_reason}")

        lines.append("\n### Reference: common guava defects")
        for name, meta in DEFECT_CATALOG.items():
            lines.append(f"- **{name}:** {meta['appearance']}")

        return status, "\n".join(lines)

    # ------------------------------------------------------------------
    def analyze(self, image_rgb: np.ndarray, max_side: int = 1000) -> PipelineOutput:
        # Phone photos are 3000-4000 px; every step here scales with pixel count, so work at <= max_side.
        hh, ww = image_rgb.shape[:2]
        if max(hh, ww) > max_side:
            sc = max_side / float(max(hh, ww))
            image_rgb = cv2.resize(image_rgb, (int(ww * sc), int(hh * sc)), interpolation=cv2.INTER_AREA)
        mask, segmented = self.segment_guava_yolov8(image_rgb)
        defects, suppressed = self.detect_defects_yolov5(image_rgb, mask)
        annotated = self.render_annotated_image(image_rgb, mask, defects, suppressed)
        color_dist, shape_metrics, shape_dist, shape_class = self.analyze_color_and_shape(
            image_rgb, mask, defects, suppressed
        )
        damage = color_dist.get("Defected / Necrotic Tissue", 0.0)
        status, report_md = self.build_diagnostic_report(
            defects, suppressed, shape_class, shape_metrics, damage
        )
        grade = _grade_from_damage(max(sum(d.area_pct for d in defects), damage))
        fig = self.create_pie_charts_figure(color_dist, shape_dist, shape_class, shape_metrics)

        return PipelineOutput(
            original_rgb=image_rgb,
            yolov8_mask=mask,
            segmented_guava_rgb=segmented,
            annotated_yolov5_rgb=annotated,
            defects=defects,
            suppressed_regions=suppressed,
            color_distribution=color_dist,
            shape_metrics=shape_metrics,
            shape_distribution=shape_dist,
            shape_classification=shape_class,
            health_status=status,
            diagnostic_markdown=report_md,
            chart_figure=fig,
            surface_damage_pct=damage,
            grade=grade,
        )


if __name__ == "__main__":
    pipe = GuavaDefectPipeline()
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    for fname in sorted(os.listdir(sample_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        bgr = cv2.imread(os.path.join(sample_dir, fname))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = pipe.analyze(rgb)
        print(f"=== {fname} ===")
        print("Status:", out.health_status)
        print("Shape Class:", out.shape_classification)
        print("Defects:", [(d.label, d.bbox, d.area_pct, d.confidence) for d in out.defects])
        print("Suppressed:", [(s.label, s.bbox, s.area_pct) for s in out.suppressed_regions])
        print("Color Dist:", out.color_distribution)
        plt.close(out.chart_figure)