# whatsapp_bot.py
import os
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, request, abort
from dotenv import load_dotenv

import csv
import datetime as dt

# import RAG pipeline
from backend import answer_with_full_rag

# ----- load Config / env -----
load_dotenv() 

#set logging in Fly
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))

#Load access tokens for WhatsApp from secrets
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
GRAPH_API_VERSION     = os.getenv("GRAPH_API_VERSION", "v22.0")
ACK_TEXT              = os.getenv("ACK_TEXT", "⏳ Thinking... I'll reply shortly!")

#Report any missing tokens
_missing = [n for n, v in [
    ("WHATSAPP_ACCESS_TOKEN", WHATSAPP_ACCESS_TOKEN),
    ("WHATSAPP_VERIFY_TOKEN", WHATSAPP_VERIFY_TOKEN),
] if not v]
if _missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(_missing)}")

#Namespace (choose which document base)
DEFAULT_NAMESPACE = os.getenv("DEFAULT_NAMESPACE", "Blood Donation")

#Messages log to this file
CSV_PATH = "/data/message_log.csv"

def log_to_csv(wa_id: str, business_phone_id: str, display_num: str,
               namespace: str, question: str, answer: str):
    """Append each Q&A to a persistent CSV file that resides in the Fly volume.
        The columns are: Time of creation (UTC), WA User ID, Business Phone ID,
        Display phone mumber (donor's), namespace (which document base), question, answer
    """
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    write_header = not os.path.exists(CSV_PATH)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "created_at_utc", "wa_user_id", "business_phone_id",
                "display_phone_number", "namespace", "question", "answer"
            ])
        writer.writerow([ts, wa_id, business_phone_id, display_num, namespace, question, answer])


def wa_id_seen(wa_id: str) -> bool:
    """Return True if previous user (wa_id) exists in CSV_PATH under the 'wa_user_id' column."""
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "wa_id" in reader.fieldnames:
                return any((row.get("wa_id") or "") == wa_id for row in reader)
    except FileNotFoundError:
        return False
    return False
    

def send_whatsapp_text(phone_number_id: str, to_number: str, body: str, timeout: int = 15) -> None:
    """
    Send a text message via WhatsApp Cloud API
    *from the exact business number that received the inbound message*.
    """
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    #payload to send back to WhatsApp
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,  # digits only (no 'whatsapp:' prefix)
        "type": "text",
        "text": {"body": body},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout) #send message
    try:
        r.raise_for_status()
        app.logger.info("Sent message via %s → %s; response: %s", phone_number_id, to_number, r.json())
    except requests.HTTPError:
        app.logger.error("WhatsApp send failed (%s): %s", r.status_code, r.text)
        raise


def _process_and_reply(user_text: str, user_wa_id: str,
                       phone_number_id: str, display_phone_number: str | None) -> None:
    """Run RAG and send reply back to the user, using the same business number."""
    
    namespace = DEFAULT_NAMESPACE

    #construct intro for first-time user
    intro_message = "Hello! I am a chat tool to help answer basic questions about blood donation. I can only see your most recent messages, so please give me appropriate context! \n\n For your question: \n "
    seen = wa_id_seen(user_wa_id) #check if previous user
    print("Phone number exists? : ", wa_id_seen(user_wa_id))

    try:
        answer, *_ = answer_with_full_rag(user_text, 5, namespace) #get answer
        if not seen:
            answer = intro_message + answer #send intro
    except Exception:
        app.logger.exception("RAG pipeline error")
        answer = "Sorry, something went wrong. Please try again."
    try:
        send_whatsapp_text(phone_number_id, user_wa_id, answer) #send text
    except Exception:
        app.logger.exception("Failed to send WhatsApp reply")
    try:
        log_to_csv(user_wa_id, phone_number_id, display_phone_number or "", namespace, user_text, answer)
    except Exception:
        app.logger.exception("Failed to log interaction to CSV")


#---- APP BUILDING ------ :

# --- Webhook verification (GET) ---
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return challenge, 200
    return abort(403)


# --- Webhook receiver (POST) ---
@app.route("/webhook", methods=["POST"])
def receive_webhook():
    data = request.get_json(silent=True) or {}
    app.logger.debug("Inbound payload: %s", data)

    #Read all entries from WhatsApp

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {}) or {}
            messages = value.get("messages", []) or []
            if not messages:
                continue  # ignore statuses-only events

            # Which business number received the message:
            meta = value.get("metadata", {}) or {}
            business_phone_id = meta.get("phone_number_id")
            display_number     = meta.get("display_phone_number")

            #Get messages and sender WhatsApp number
            msg   = messages[0]
            wa_id = msg.get("from")  # user's WhatsApp number (digits)
            text  = (msg.get("text") or {}).get("body")

            if business_phone_id and wa_id and text:
                # 1) Immediate response to user saying THINKING from the same business number
                try:
                    send_whatsapp_text(business_phone_id, wa_id, ACK_TEXT, timeout=5)
                except Exception:
                    app.logger.exception("Failed to send ack")

                # 2) Final answer
                executor.submit(_process_and_reply, text, wa_id, business_phone_id, display_number)

    return "EVENT_RECEIVED", 200



if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
