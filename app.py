import os, json, io, smtplib
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from email.message import EmailMessage
from PIL import Image
import cv2, numpy as np
from dotenv import load_dotenv

from plate_detector import extract_plate_text, normalize_plate_string
from compliance import evaluate_vehicle

load_dotenv()

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
UPLOAD_FOLDER = "uploads"
LOG_FILE = "scan_log.json"

SMTP_USER = os.getenv("ALERT_SMTP_USER")
SMTP_PASS = os.getenv("ALERT_SMTP_PASS")
SMTP_HOST = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("ALERT_SMTP_PORT", 587))
RECIPIENTS = os.getenv("ALERT_RECIPIENTS", "").split(",")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def send_alert_email(plate, make, model, year, age):
    if not SMTP_USER or not SMTP_PASS or not RECIPIENTS:
        return

    msg = EmailMessage()
    msg["Subject"] = f"BANNED Vehicle Detected: {plate}"
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(RECIPIENTS)
    msg.set_content(f"""
    BANNED VEHICLE DETECTED

    License Plate: {plate}
    Make & Model: {make} {model}
    Registration Year: {year}
    Vehicle Age: {age} years

    This vehicle exceeds the 15-year age limit.
    """)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception:
        pass

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logger")
def logger():
    return render_template("logger.html")

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "invalid file type"}), 400

    fname = secure_filename(file.filename)
    data = file.read()

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return jsonify({"error": "unable to read image"}), 400

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    plate_raw = extract_plate_text(cv_img) or ""
    plate_normalized = normalize_plate_string(plate_raw) or ""

    result = evaluate_vehicle(plate_normalized)

    response = {
        "plate_raw": plate_raw,
        "plate_normalized": plate_normalized,
        "make": result.get("make"),
        "model": result.get("model"),
        "registration_year": result.get("registration_year"),
        "age": result.get("age"),
        "banned": result.get("banned"),
        "error": result.get("error"),
    }

    if response["banned"]:
        send_alert_email(
            plate=response["plate_normalized"],
            make=response.get("make"),
            model=response.get("model"),
            year=response.get("registration_year"),
            age=response.get("age"),
        )

    try:
        with open(os.path.join(app.config["UPLOAD_FOLDER"], fname), "wb") as f:
            f.write(data)
        saved_image = fname
    except Exception:
        saved_image = None

    try:
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            try:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
            except json.JSONDecodeError:
                logs = []
            logs.append({**response, "image": saved_image})
            f.seek(0)
            json.dump(logs, f, indent=2)
            f.truncate()
    except Exception:
        pass

    return jsonify(response)

@app.route("/api/logs")
def get_logs():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                logs = []
    except Exception:
        logs = []
    return jsonify(logs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
