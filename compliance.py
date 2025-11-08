
import requests
import xml.etree.ElementTree as ET
import datetime as dt
import re
import os
from dotenv import load_dotenv
load_dotenv()
REGCHECK_BASE_URL = os.getenv(
    "REGCHECK_BASE_URL",
    "http://www.regcheck.org.uk/api/reg.asmx/CheckIndia"
)
REGCHECK_USERNAME = os.getenv("REGCHECK_USERNAME", "")
REQUEST_TIMEOUT = 10           # Seconds

def _extract_year(text):
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", str(text))
    return int(match.group(0)) if match else None

def get_vehicle_info(reg_no: str):
    params = {
        "RegistrationNumber": reg_no,
        "username": REGCHECK_USERNAME
    }

    try:
        response = requests.get(REGCHECK_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        root = ET.fromstring(response.text)

        ns = {"ns": "http://regcheck.org.uk"}

        reg_year = root.findtext(".//ns:RegistrationYear", namespaces=ns)
        make = root.findtext(".//ns:CarMake/ns:CurrentTextValue", namespaces=ns)
        model = root.findtext(".//ns:CarModel", namespaces=ns)

        reg_year = _extract_year(reg_year)

        return {
            "ok": True,
            "make": make or "",
            "model": model or "",
            "registration_year": reg_year
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

def check_15_year_rule(reg_year):
    if not reg_year:
        return False, None

    current_year = dt.datetime.now().year
    age = current_year - reg_year
    return (age >= 15), age

def evaluate_vehicle(reg_no: str):
    data = get_vehicle_info(reg_no)

    result = {
        "plate": reg_no,
        "make": data.get("make", ""),
        "model": data.get("model", ""),
        "registration_year": data.get("registration_year"),
        "age": None,
        "banned": False
    }

    if data.get("registration_year"):
        banned, age = check_15_year_rule(data["registration_year"])
        result["banned"] = banned
        result["age"] = age

    return result
