# plate_detector.py
import re
import cv2
import numpy as np
from typing import List, Tuple

import easyocr

# --- YOLOv8 for plate detection ---
from ultralytics import YOLO

# --- PaddleOCR for text reading ---
from paddleocr import PaddleOCR

# ---------------------------
# Lazy singletons (load once)
# ---------------------------
_yolo_model = None
_ocr_reader = None

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        # Lightweight license-plate model from the hub; downloads on first run
        _yolo_model = YOLO("license_plate_detector.pt")
    return _yolo_model

def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        # English is enough for Indian alphanumeric plates
        _ocr_reader = PaddleOCR(lang='en', use_angle_cls=True)
    return _ocr_reader

# ---------------------------
# Helpers
# ---------------------------
ALNUM = re.compile(r'[^A-Z0-9]')

def normalize_plate_string(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    return ALNUM.sub('', s)

def _post_correct(text: str) -> str:
    """
    Correct common OCR confusions for Indian plates.
    """
    repl = {
        'O': '0',
        'I': '1',
        'L': '1',
        'B': '8',
        'S': '5',
        'Z': '2',
        'Q': '0'
    }
    return ''.join(repl.get(c, c) for c in text)

# Indian plate format heuristics (very simple):
# e.g., TN09AB1234, KA01BB0001, MH12DE1432, BH12AB1234
_PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}$')

def _score_candidate(s: str) -> int:
    """
    Score by (format match, length, digits+letters mix).
    Higher is better.
    """
    score = 0
    if _PLATE_RE.match(s):
        score += 50
    score += min(len(s), 10)          # favor longer up to ~10
    score += sum(c.isdigit() for c in s)
    score += sum(c.isalpha() for c in s)
    return score

# ---------------------------
# Image preprocessing
# ---------------------------
def _deskew(img_bgr: np.ndarray) -> np.ndarray:
    """
    Deskew using minAreaRect angle on edges.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Edge map to find text contours more robustly
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.size == 0:
        return img_bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def _enhance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance contrast using CLAHE on L-channel, then binarize.
    Returns a single-channel image good for OCR.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    # Slight blur, then Otsu
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Light dilation to thicken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.dilate(th, kernel, iterations=1)
    return th

def _smart_crop(img_bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int], pad_ratio: float = 0.12) -> np.ndarray:
    """
    Crop with a bit of padding around the YOLO box.
    """
    x1, y1, x2, y2 = box_xyxy
    h, w = img_bgr.shape[:2]
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    return img_bgr[y1:y2, x1:x2]

# ---------------------------
# Detection + OCR pipeline
# ---------------------------
def _detect_plates_yolo(image_bgr: np.ndarray, conf: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of boxes in xyxy format (x1,y1,x2,y2).
    """
    model = _get_yolo()
    results = model.predict(image_bgr, conf=conf, verbose=False)[0]
    boxes = []
    if results.boxes is not None:
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes.append((x1, y1, x2, y2))
    return boxes

# ---------------------------
# EasyOCR for final recognition (Windows stable)
# ---------------------------
_easy_reader = None

def _ocr_easy(image_bgr_or_gray: np.ndarray) -> str:
    global _easy_reader

    if _easy_reader is None:
        _easy_reader = easyocr.Reader(['en'])  # supports English alphanumeric

    # ✅ Always convert to RGB for OCR
    if len(image_bgr_or_gray.shape) == 2:
        img = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2RGB)

    results = _easy_reader.readtext(img, detail=0)
    
    # Merge results
    text = "".join(results)
    text = normalize_plate_string(text)
    text = _post_correct(text)
    return text

def extract_plate_text(image_bgr: np.ndarray) -> str:
    """
    Full pipeline:
      1) YOLO detect plate boxes
      2) For each box: crop -> deskew -> enhance -> OCR
      3) If no boxes, OCR whole image as fallback
      4) Pick best candidate by score
    """
    candidates = []

    boxes = _detect_plates_yolo(image_bgr, conf=0.25)
    print("YOLO detection boxes:", boxes)

    if boxes:
        for (x1, y1, x2, y2) in boxes:
            crop = _smart_crop(image_bgr, (x1, y1, x2, y2))
            cv2.imwrite("debug_crop.jpg", crop)
            print("Saved crop to debug_crop.jpg")
            if crop.size == 0:
                continue

            # Resize crop to ~width 500px for better OCR
            h, w = crop.shape[:2]
            if w < 500:
                scale = 500 / w
                crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            deskewed = _deskew(crop)
            prepped = _enhance(deskewed)
            cv2.imwrite("debug_prepped.jpg", prepped)
            print("Saved processed image to debug_prepped.jpg")
            text = _ocr_easy(prepped)
            if text:
                candidates.append(text)

    # Fallback: OCR entire image
    if not candidates:
        # Resize entire image smaller or larger to ~900px width
        h, w = image_bgr.shape[:2]
        scale = 900 / w if w != 0 else 1.0
        img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        prepped = _enhance(_deskew(img))
        text = _ocr_easy(prepped)
        if text:
            candidates.append(text)

    if not candidates:
        return ""

    # Pick best candidate by heuristic score
    candidates = sorted(candidates, key=_score_candidate, reverse=True)
    return candidates[0]
