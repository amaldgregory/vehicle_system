# app.py
import os
import json
from flask import Flask, request, render_template, jsonify
from plate_detector import extract_plate_text, normalize_plate_string
from email_sender import send_alert_email
from werkzeug.utils import secure_filename
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()


ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
UPLOAD_FOLDER = "uploads"
LOG_FILE = "scan_log.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load banned list
with open("banned_plates.json", "r") as f:
    banned_data = json.load(f)
BANNED_SET = set([normalize_plate_string(x) for x in banned_data.get("banned", [])])

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)
        
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logger")
def logger():
    return render_template("logger.html")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if file and allowed_file(file.filename):
        fname = secure_filename(file.filename)
        data = file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # Convert to OpenCV image (BGR)
        import cv2
        import numpy as np

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


        plate = extract_plate_text(cv_img)
        normalized = normalize_plate_string(plate)

        response = {"plate_raw": plate, "plate_normalized": normalized, "banned": False}

        if normalized and normalized in BANNED_SET:
            response["banned"] = True
            # Send alert email (non-blocking option: here we call directly; for scale, consider background jobs)
            try:
                send_alert_email(normalized)
                response["email_sent"] = True
            except Exception as e:
                response["email_sent"] = False
                response["email_error"] = str(e)

        # Optionally save uploaded image
        with open(os.path.join(app.config["UPLOAD_FOLDER"], fname), "wb") as f:
            f.write(data)

        with open(LOG_FILE, "r+") as f:
            logs = json.load(f)
            logs.append({
                "plate_raw": plate,
                "plate_normalized": normalized,
                "banned": response["banned"],
                "image": fname
            })
            f.seek(0)
            json.dump(logs, f, indent=2)

        return jsonify(response)

    return jsonify({"error": "invalid file type"}), 400

@app.route("/api/logs")
def get_logs():
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)
    return jsonify(logs)

if __name__ == "__main__":
    # For dev only: set debug True
    app.run(host="0.0.0.0", port=5000, debug=True)
