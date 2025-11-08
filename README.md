License Plate Detection & 15-Year Compliance Checker 
This project is a web-based system that: 
- Detects vehicle license plates from uploaded images using OCR (Tesseract/EasyOCR + OpenCV + YOLO). 
- Extracts and normalizes the plate number. 
- Calls a vehicle details API (RegCheck) to get registration year, make, and model. 
- Checks if the vehicle is older than 15 years → If yes, marks it as **BANNED**. 
- Sends an email alert when a banned vehicle is detected. -
- Logs every scan to a JSON file and displays it in a dashboard.

NOW before you run our project , make sure of these PREREQUISITES :-

1) Make sure to install all the required modules using
           pip install -r requirements.txt
2) Make an account and get your username from [RegCheck](https://www.carregistrationapi.in) for the API.
3) We have used a OCR engine called **Tesseract** to enhance our OCR models , you would have to install the .exe file from
           [<Tesseract.exe>](https://github.com/UB-Mannheim/tesseract/wiki)
   Also make sure to add the path of tesseract.exe to your ENVIRONMENT PATH VARIABLE or 
   We recommend making a .env file with the variables
    REGCHECK_USERNAME = "Username for the RegCheck API Website - which you will need to retrieve make,model and registration year
    ALERT_SMTP_USER = "Sender email-id"
    ALERT_SMTP_PASS = "App Password for Sender email-id"
    ALERT_RECIPIENTS = "Recipien ts emai-id"
    ALERT_SMTP_HOST = smtp.gmail.com
    ALERT_SMTP_PORT = 587
    TESSERACT_PATH = "PATH where tesseract.exe is installed"

