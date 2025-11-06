# email_sender.py
import os
import smtplib
from email.message import EmailMessage

def send_alert_email(plate: str, smtp_user=None, smtp_pass=None, recipients=None):
    """
    Send an email alert that a banned plate was detected.
    Expects SMTP credentials (e.g. Gmail app password).
    """
    smtp_user = smtp_user or os.environ.get("ALERT_SMTP_USER")
    smtp_pass = smtp_pass or os.environ.get("ALERT_SMTP_PASS")
    recipients = recipients or os.environ.get("ALERT_RECIPIENTS", "").split(",")

    if not smtp_user or not smtp_pass or not recipients or recipients == [""]:
        raise RuntimeError("SMTP user/pass or recipients not configured. Set environment vars ALERT_SMTP_USER, ALERT_SMTP_PASS, ALERT_RECIPIENTS")

    msg = EmailMessage()
    msg["Subject"] = f"ALERT: Banned license plate detected — {plate}"
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    body = f"A banned license plate has been detected:\n\nPlate: {plate}\n\nThis is an automated alert."
    msg.set_content(body)

    # Example: Gmail SMTP (smtp.gmail.com:587)
    smtp_host = os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("ALERT_SMTP_PORT", 587))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
