# plate_detector.py
import cv2
import pytesseract
import numpy as np
import re
from PIL import Image

# If tesseract binary is not on PATH, uncomment and set:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CASCADE_PATH = "cascades/haarcascade_russian_plate_number.xml"

def normalize_plate_string(s: str) -> str:
    """Uppercase and keep only alphanumeric characters (A-Z,0-9)."""
    if not s:
        return ""
    s = s.upper()
    return re.sub(r'[^A-Z0-9]', '', s)

def ocr_image_pil(pil_image: Image.Image) -> str:
    """Run pytesseract on a PIL image with some config tuned for alphanumeric."""
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(pil_image, config=config)
    return normalize_plate_string(text)

def preprocess_for_ocr(img):
    """Preprocess OpenCV image for better OCR: grayscale, bilateral denoise, adaptive threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize to increase OCR accuracy
    h, w = gray.shape
    scale = max(1, 400 / w)  # aim for width ~400px
    if scale != 1:
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 8)
    return th

def detect_plate_regions(image) -> list:
    """Return list of bounding boxes (x,y,w,h) for detected plate regions using Haar cascade."""
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detectMultiScale parameters may need tuning
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,20))
    return plates.tolist() if isinstance(plates, np.ndarray) else []

def extract_plate_text(image) -> str:
    """
    Try to detect plate regions, OCR them, and return the best candidate string.
    Falls back to OCR on the whole image if no region found.
    """
    boxes = detect_plate_regions(image)
    candidates = []

    if boxes:
        for (x, y, w, h) in boxes:
            # expand box a bit
            pad_x = int(w * 0.08)
            pad_y = int(h * 0.12)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(image.shape[1], x + w + pad_x)
            y1 = min(image.shape[0], y + h + pad_y)
            crop = image[y0:y1, x0:x1]
            proc = preprocess_for_ocr(crop)
            pil = Image.fromarray(proc)
            text = ocr_image_pil(pil)
            if text:
                candidates.append(text)

    # fallback: OCR whole image
    if not candidates:
        proc = preprocess_for_ocr(image)
        text = ocr_image_pil(Image.fromarray(proc))
        if text:
            candidates.append(text)

    # Choose the best candidate — longest plausible alphanumeric string
    if candidates:
        candidates = sorted(candidates, key=lambda s: (-len(s), s))
        return candidates[0]
    return ""
