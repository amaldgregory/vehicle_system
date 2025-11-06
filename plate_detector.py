# plate_detector.py
import re
import cv2
import numpy as np
from typing import List, Tuple

import easyocr
from ultralytics import YOLO  # YOLOv8 detector

# ---------------------------
# Lazy singletons (load once)
# ---------------------------
_yolo_model = None
_easy_reader = None

# ==== Config ====
DESKEW = False           # keep False as requested (no rotation)
PAD_RATIO = 0.18         # padding around YOLO box before OCR
TARGET_WIDTH = 350       # resize plate crops to about this width
MAX_HEIGHT = 200         # cap height; code preserves aspect ratio
USE_BINARY_FALLBACK = True
MIN_PLATE_LEN = 6        # if OCR shorter than this, try fallback
SAVE_DEBUG = True        # writes debug images next to app.py
YOLO_WEIGHTS = "license_plate_detector.pt"  # change if you named it differently
# =================

# ---------- Helpers ----------
ALNUM = re.compile(r'[^A-Z0-9]')

def normalize_plate_string(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    return ALNUM.sub('', s)

def _post_correct(text: str) -> str:
    # common confusions on Indian plates
    repl = {'O': '0','I': '1','L': '1','B': '8','S': '5','Z': '2','Q': '0'}
    return ''.join(repl.get(c, c) for c in text)

# Indian plate format heuristic
_PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}$')

def _score_candidate(s: str) -> int:
    score = 0
    if _PLATE_RE.match(s):
        score += 50
    score += min(len(s), 10)
    score += sum(c.isdigit() for c in s)
    score += sum(c.isalpha() for c in s)
    return score

# ---------- Models ----------
def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_WEIGHTS)
        print(f"YOLO model loaded: {YOLO_WEIGHTS}")
    return _yolo_model

def _get_easyocr():
    global _easy_reader
    if _easy_reader is None:
        # gpu=False on Windows unless you have CUDA set up
        _easy_reader = easyocr.Reader(
            ['en'], gpu=False, recog_network='latin_g2', verbose=False
        )
    return _easy_reader

# ---------- Geometry ----------
def _smart_crop(img_bgr: np.ndarray, box_xyxy: tuple[int, int, int, int], pad_ratio: float = PAD_RATIO) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    H, W = img_bgr.shape[:2]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pw = int(w * pad_ratio)
    ph = int(h * pad_ratio)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw)
    y2 = min(H, y2 + ph)
    return img_bgr[y1:y2, x1:x2]

def _deskew_rotate_bound(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.size == 0:
        return img_bgr
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - (w / 2)
    M[1, 2] += (nH / 2) - (h / 2)
    return cv2.warpAffine(img_bgr, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# ---------- Preprocess ----------
def _resize_preserve_ar(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if w >= TARGET_WIDTH:
        return crop_bgr
    scale = TARGET_WIDTH / max(1, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if new_h > MAX_HEIGHT:
        # keep aspect ratio when capping height
        scale2 = MAX_HEIGHT / max(1, new_h)
        new_h = MAX_HEIGHT
        new_w = int(round(new_w * scale2))
    return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def _light_sharpen(img_bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img_bgr, -1, kernel)

def _preprocess_soft(img_bgr: np.ndarray) -> np.ndarray:
    # grayscale -> normalize -> back to RGB (EasyOCR expects RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def _preprocess_binary(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

# ---------- Detection ----------
def _detect_plates_yolo(image_bgr: np.ndarray, conf: float = 0.20) -> List[Tuple[int, int, int, int]]:
    model = _get_yolo()
    results = model.predict(image_bgr, conf=conf, verbose=False)[0]
    boxes = []
    if results.boxes is not None:
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes.append((x1, y1, x2, y2))
    return boxes

# ---------- OCR ----------
def _ocr_easy(image_rgb: np.ndarray) -> str:
    reader = _get_easyocr()

    # EasyOCR tune: strong for license plates
    results = reader.readtext(
        image_rgb,
        detail=0,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        text_threshold=0.4,   # increase if spurious; decrease if missing
        low_text=0.3,
        link_threshold=0.4,
        mag_ratio=1.5,       # slight internal upscale for detector
        slope_ths=0.2,
        ycenter_ths=0.5,
        width_ths=0.7,
        height_ths=0.6,
        decoder='greedy',
        beamWidth=5,
        paragraph=False
    )

    text = "".join(results)
    text = normalize_plate_string(text)
    text = _post_correct(text)
    return text

# ---------- Full pipeline ----------
def extract_plate_text(image_bgr: np.ndarray) -> str:
    candidates = []

    boxes = _detect_plates_yolo(image_bgr, conf=0.20)
    print("YOLO detection boxes:", boxes)

    if boxes:
        for (x1, y1, x2, y2) in boxes:
            crop = _smart_crop(image_bgr, (x1, y1, x2, y2))
            if SAVE_DEBUG:
                cv2.imwrite("debug_crop.jpg", crop)
            if crop.size == 0:
                continue

            # Resize without distortion + light sharpen
            crop = _resize_preserve_ar(crop)
            crop = _light_sharpen(crop)

            # Optional rotation (disabled by default)
            proc_base = _deskew_rotate_bound(crop) if DESKEW else crop

            # Soft path first
            soft_rgb = _preprocess_soft(proc_base)
            if SAVE_DEBUG:
                cv2.imwrite("debug_soft.jpg", soft_rgb)
            print("Soft OCR input shape:", soft_rgb.shape)

            text = _ocr_easy(soft_rgb)

            # Binary fallback
            if (not text) or (len(text) < MIN_PLATE_LEN and USE_BINARY_FALLBACK):
                hard_rgb = _preprocess_binary(proc_base)
                if SAVE_DEBUG:
                    cv2.imwrite("debug_binary.jpg", hard_rgb)
                print("Binary OCR input shape:", hard_rgb.shape)
                text2 = _ocr_easy(hard_rgb)
                if not text and text2:
                    text = text2

            if text:
                candidates.append(text)

    # Whole-image fallback (rare)
    if not candidates:
        h, w = image_bgr.shape[:2]
        scale = 900 / max(1, w)
        img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        img = _light_sharpen(img)
        proc_base = _deskew_rotate_bound(img) if DESKEW else img

        soft_rgb = _preprocess_soft(proc_base)
        text = _ocr_easy(soft_rgb)

        if (not text) or (len(text) < MIN_PLATE_LEN):
            hard_rgb = _preprocess_binary(proc_base)
            text2 = _ocr_easy(hard_rgb)
            if not text and text2:
                text = text2

        if text:
            candidates.append(text)

    if not candidates:
        return ""

    candidates = sorted(candidates, key=_score_candidate, reverse=True)
    return candidates[0]
