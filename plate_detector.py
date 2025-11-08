
# plate_detector.py
import re
import cv2
import numpy as np
from typing import List, Tuple
import os
from dotenv import load_dotenv
import easyocr
import pytesseract
from ultralytics import YOLO  # YOLOv8 detector
load_dotenv()
tesseract_cmd = os.getenv("TESSERACT_PATH")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


_yolo_model = None
_easy_reader = None

DESKEW = False             
PAD_RATIO = 0.25            
BORDER = 6                   
TARGET_WIDTH = 420           
MAX_HEIGHT = 240
SAVE_DEBUG = True
YOLO_WEIGHTS = "license_plate_detector.pt"
# =====================================
PLATE_FMT_STRICT = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')  
BH_FMT_STRICT    = re.compile(r'^\d{2}BH\d{4}[A-Z]{2}$') 
ALNUM = re.compile(r'[^A-Z0-9]')
PLATE_FMT = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}$')
ROLES_10 = [False, False, True, True, False, False, True, True, True, True]
LETTER_FOR_DIGIT = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
DIGIT_FOR_LETTER  = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'Q': '0'}

_TESS_LINE = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_TESS_CHAR = "--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _coerce_by_roles(s: str, roles: list[bool]) -> str:
    """
    roles: False -> expect LETTER, True -> expect DIGIT
    Coerce only up to len(roles); leave any extra chars untouched (already alnum+upper).
    """
    out = list(s)
    upto = min(len(out), len(roles))
    for i in range(upto):
        ch = out[i]
        out[i] = DIGIT_FOR_LETTER.get(ch, ch) if roles[i] else LETTER_FOR_DIGIT.get(ch, ch)
    return "".join(out)

ROLES_TWO_SERIES = [False, False, True, True, False, False, True, True, True, True]  
ROLES_ONE_SERIES = [False, False, True, True, False,        True, True, True, True]  

def normalize_plate_string(s: str) -> str:
    """
    1) Clean -> uppercase, keep only A-Z0-9.
    2) Try BH format:  NN BH NNNN AA  (with tolerant parse, strict output)
    3) Try classic format with tolerant parse, then fix:
         - series block -> letters only
         - number block -> digits only
    4) Fallback to role-based coercion (supports 1- or 2-letter series).
    """
    if not s:
        return ""
    s = ALNUM.sub('', s.upper())
    if not s:
        return ""

    m = re.match(r'^(\d{2})B[H4N]([A-Z0-9]{4})([A-Z0-9]{2})$', s)
    if m:
        yy, mid, tail = m.groups()
        # digits only in middle 4
        mid  = ''.join(DIGIT_FOR_LETTER.get(c, c) for c in mid)
        # letters only in last 2
        tail = ''.join(LETTER_FOR_DIGIT.get(c, c) for c in tail)
        cand = f"{yy}BH{mid}{tail}"
        if BH_FMT_STRICT.match(cand):
            return cand

    m = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z0-9]{1,2})([A-Z0-9]{3,4})$', s)
    if m:
        state, dist, series, number = m.groups()
        series = ''.join(LETTER_FOR_DIGIT.get(c, c) for c in series)
        number = ''.join(DIGIT_FOR_LETTER.get(c, c) for c in number)
        cand = f"{state}{dist}{series}{number}"
        if PLATE_FMT_STRICT.match(cand):
            return cand

    cand_two = _coerce_by_roles(s, ROLES_TWO_SERIES)[:10]
    cand_one = _coerce_by_roles(s, ROLES_ONE_SERIES)[:9]
    if PLATE_FMT_STRICT.match(cand_two): return cand_two
    if PLATE_FMT_STRICT.match(cand_one): return cand_one

    def _score_loose(t: str) -> int:
        return (9 <= len(t) <= 10) + sum(c.isalpha() for c in t) + sum(c.isdigit() for c in t)
    return cand_two if _score_loose(cand_two) >= _score_loose(cand_one) else cand_one

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_WEIGHTS)
        print(f"YOLO model loaded: {YOLO_WEIGHTS}")
    return _yolo_model

def _get_easyocr():
    global _easy_reader
    if _easy_reader is None:
        _easy_reader = easyocr.Reader(['en'], gpu=False, recog_network='latin_g2', verbose=False)
    return _easy_reader

def _smart_crop(img_bgr: np.ndarray, box_xyxy: Tuple[int,int,int,int]) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    H, W = img_bgr.shape[:2]
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    pw = int(w * PAD_RATIO); ph = int(h * PAD_RATIO)
    x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw); y2 = min(H, y2 + ph)
    crop = img_bgr[y1:y2, x1:x2]
    if BORDER:
        crop = cv2.copyMakeBorder(crop, BORDER, BORDER, BORDER+2, BORDER+2, cv2.BORDER_REPLICATE)
    return crop

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
    nW = int((h * sin) + (w * cos)); nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - (w / 2); M[1, 2] += (nH / 2) - (h / 2)
    return cv2.warpAffine(img_bgr, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _rectify_by_contour(plate_bgr: np.ndarray) -> np.ndarray:
    """Find a rectangular contour and perspective-warp the plate. Falls back to input."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 7)
    cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area; best = approx
    if best is None:
        return plate_bgr
    quad = best.reshape(-1, 2).astype(np.float32)

    def order(pts):
        s = pts.sum(axis=1); diff = np.diff(pts, axis=1).ravel()
        return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                         pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)
    quad = order(quad)

    w = int(max(np.linalg.norm(quad[0]-quad[1]), np.linalg.norm(quad[2]-quad[3])))
    h = int(max(np.linalg.norm(quad[0]-quad[3]), np.linalg.norm(quad[1]-quad[2])))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(plate_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped

# ---------------- Preprocess ----------------
def _resize_preserve_ar(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    scale = max(TARGET_WIDTH / max(1, w), 1.0)
    new_w = int(round(w * scale)); new_h = int(round(h * scale))
    if new_h > MAX_HEIGHT:
        scale2 = MAX_HEIGHT / max(1, new_h)
        new_h = MAX_HEIGHT; new_w = int(round(new_w * scale2))
    return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def _unsharp(img):
    g = cv2.GaussianBlur(img, (0,0), 1.0)
    return cv2.addWeighted(img, 1.6, g, -0.6, 0)

def _prep_soft(img_bgr: np.ndarray) -> np.ndarray:
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x = cv2.bilateralFilter(x, 7, 75, 75)
    x = cv2.createCLAHE(2.0, (8,8)).apply(x)
    x = _unsharp(x)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

def _prep_bin(img_bgr: np.ndarray) -> np.ndarray:
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x = cv2.createCLAHE(2.0, (8,8)).apply(x)
    x = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 29, 5)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

# ---------------- Detection ----------------
def _detect_plates_yolo(image_bgr: np.ndarray, conf: float = 0.20) -> List[Tuple[int,int,int,int]]:
    model = _get_yolo()
    results = model.predict(image_bgr, conf=conf, verbose=False)[0]
    boxes = []
    if results.boxes is not None:
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes.append((x1, y1, x2, y2))
    return boxes

# ---------------- OCR ----------------
def _ocr_easy(image_rgb: np.ndarray) -> Tuple[str, float]:
    reader = _get_easyocr()
    res = reader.readtext(
        image_rgb, detail=1,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        text_threshold=0.4, low_text=0.3, link_threshold=0.4,
        mag_ratio=1.8, slope_ths=0.2, ycenter_ths=0.5, width_ths=0.7, height_ths=0.6,
        decoder='greedy', beamWidth=5, paragraph=False
    )
    texts, confs = [], []
    for *_, text, conf in res:
        if text:
            t = ALNUM.sub('', text.upper())
            if t:
                texts.append(t)
                if conf is not None and conf >= 0:
                    confs.append(float(conf)*100.0)
    merged = normalize_plate_string("".join(texts))
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return merged, mean_conf

def _ocr_tess_line(image_rgb: np.ndarray) -> Tuple[str, float]:
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT, config=_TESS_LINE)
    tokens, confs = [], []
    for txt, conf in zip(data["text"], data["conf"]):
        if txt and txt.strip() and conf != '-1':
            t = ALNUM.sub('', txt.upper())
            if t:
                tokens.append(t)
                try: confs.append(float(conf))
                except: pass
    merged = normalize_plate_string("".join(tokens))
    return merged, (float(np.mean(confs)) if confs else 0.0)

def _ocr_tess_charwise(image_rgb: np.ndarray) -> Tuple[str, float]:
    """Segment characters left→right and run Tesseract psm10 per glyph."""
    g = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)

    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h < 0.35*g.shape[0] or w*h < 40:   
            continue
        boxes.append((x,y,w,h))
    boxes = sorted(boxes, key=lambda b: b[0])

    chars, confs = [], []
    for (x,y,w,h) in boxes:
        ch_img = g[max(0,y-2):y+h+2, max(0,x-2):x+w+2]
        data = pytesseract.image_to_data(ch_img, output_type=pytesseract.Output.DICT, config=_TESS_CHAR)
        best_t, best_c = "", -1
        for txt, conf in zip(data["text"], data["conf"]):
            if txt and txt.strip() and conf != '-1':
                t = ALNUM.sub('', txt.upper())
                if t and float(conf) > best_c:
                    best_t, best_c = t, float(conf)
        if best_t:
            chars.append(best_t); confs.append(best_c)
    merged = normalize_plate_string("".join(chars))
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return merged, mean_conf

# ---------------- Scoring ----------------
def _score(text: str, conf: float) -> float:
    score = 0.0
    if PLATE_FMT.match(text): score += 60.0
    score += min(len(text), 10) * 2.0
    score += sum(c.isdigit() for c in text) + sum(c.isalpha() for c in text)
    score += 0.5 * conf
    return score

def _try_all(proc_base_bgr: np.ndarray) -> Tuple[str, float]:
    rect = _rectify_by_contour(proc_base_bgr)
    rect = _resize_preserve_ar(rect)

    soft = _prep_soft(rect); binv = _prep_bin(rect)

    variants = []
    for img in (soft, binv):
        t1, c1 = _ocr_easy(img);         variants.append((t1,c1))
        t2, c2 = _ocr_tess_line(img);    variants.append((t2,c2))
        t3, c3 = _ocr_tess_charwise(img);variants.append((t3,c3))

    if SAVE_DEBUG:
        cv2.imwrite("debug_rect.jpg", rect)
        cv2.imwrite("debug_soft.jpg", cv2.cvtColor(soft, cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_bin.jpg",  cv2.cvtColor(binv, cv2.COLOR_RGB2BGR))

    return max(variants, key=lambda tc: _score(tc[0], tc[1]), default=("",0.0))

# ---------------- Pipeline ----------------
def _light_sharpen(img_bgr: np.ndarray) -> np.ndarray:
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img_bgr, -1, k)

def _detect_plates(image_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    return _detect_plates_yolo(image_bgr, conf=0.20)

def extract_plate_text_and_conf(image_bgr: np.ndarray) -> Tuple[str, float]:
    cands: List[Tuple[str, float]] = []
    boxes = _detect_plates(image_bgr)
    print("YOLO detection boxes:", boxes)

    if boxes:
        for (x1,y1,x2,y2) in boxes:
            crop = _smart_crop(image_bgr, (x1,y1,x2,y2))
            if SAVE_DEBUG: cv2.imwrite("debug_crop.jpg", crop)
            if crop.size == 0: continue

            base = _light_sharpen(crop)
            if DESKEW: base = _deskew_rotate_bound(base)

            text, conf = _try_all(base)
            if text: cands.append((text, conf))

    if not cands:
        h,w = image_bgr.shape[:2]
        scale = 900 / max(1,w)
        img = cv2.resize(image_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        img = _light_sharpen(img)
        base = _deskew_rotate_bound(img) if DESKEW else img
        text, conf = _try_all(base)
        if text: cands.append((text, conf))

    if not cands:
        return "", 0.0
    return max(cands, key=lambda tc: _score(tc[0], tc[1]))

def extract_plate_text(image_bgr: np.ndarray) -> str:
    text, _ = extract_plate_text_and_conf(image_bgr)
    return text
