# whatsapp_bot.py
import os
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, request, abort
from dotenv import load_dotenv

import csv
import datetime as dt
from collections import deque

# Your RAG pipeline
from backend import answer_with_full_rag

# ----- Config / env -----
load_dotenv()  # no-op on Fly; useful for local runs

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))

WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
GRAPH_API_VERSION     = os.getenv("GRAPH_API_VERSION", "v22.0")
ACK_TEXT              = os.getenv("ACK_TEXT", "⏳ Thinking... I'll reply shortly!")

_missing = [n for n, v in [
    ("WHATSAPP_ACCESS_TOKEN", WHATSAPP_ACCESS_TOKEN),
    ("WHATSAPP_VERIFY_TOKEN", WHATSAPP_VERIFY_TOKEN),
] if not v]
if _missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(_missing)}")

# Optional: map a business phone_number_id (or display number digits) to a namespace/bot
NAMESPACE_MAP = {
    # "734690309731285": "Blood Donation",  # example: phone_number_id -> namespace
    # "15551515454": "Pregnancy",           # example: display number digits -> namespace
}
DEFAULT_NAMESPACE = os.getenv("DEFAULT_NAMESPACE", "Blood Donation")

CSV_PATH = "/data/message_log.csv"

def fetch_recent_from_csv(wa_id: str, business_phone_id: str, limit: int = 8) -> list[tuple[str, str]]:
    """
    Returns [(role, text), ...] oldest->newest for this wa_id+biz_number.
    CSV columns are: created_at_utc, wa_user_id, business_phone_id,
                     display_phone_number, namespace, question, answer
    """
    path = CSV_PATH
    if not os.path.exists(path):
        return []

    window = deque(maxlen=limit * 2)  # keep user+assistant lines
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 7:
                continue
            if row[1] != wa_id or row[2] != business_phone_id:
                continue
            q, a = row[5].strip(), row[6].strip()  # question, answer
            if q:
                window.append(("user", q))
            if a:
                window.append(("assistant", a))
    return list(window)[-limit:]


def log_to_csv(wa_id: str, business_phone_id: str, display_num: str,
               namespace: str, question: str, answer: str):
    """Append each Q&A to a persistent CSV file in the Fly volume."""
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


def _resolve_namespace(phone_number_id: str, display_phone_number: str | None) -> str:
    if phone_number_id in NAMESPACE_MAP:
        return NAMESPACE_MAP[phone_number_id]
    if display_phone_number:
        digits = "".join(ch for ch in display_phone_number if ch.isdigit())
        if digits in NAMESPACE_MAP:
            return NAMESPACE_MAP[digits]
    return DEFAULT_NAMESPACE


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
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,  # digits only (no 'whatsapp:' prefix)
        "type": "text",
        "text": {"body": body},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        r.raise_for_status()
        app.logger.info("Sent message via %s → %s; response: %s", phone_number_id, to_number, r.json())
    except requests.HTTPError:
        app.logger.error("WhatsApp send failed (%s): %s", r.status_code, r.text)
        # You can add a specific check for 131047 here if desired.
        raise


def _process_and_reply(user_text: str, user_wa_id: str,
                       phone_number_id: str, display_phone_number: str | None) -> None:
    """Run RAG and send reply back to the user, using the same business number."""
    namespace = _resolve_namespace(phone_number_id, display_phone_number)

    # Pull a small rolling window of recent context for this user+number
    history = fetch_recent_from_csv(user_wa_id, phone_number_id, limit=8)

    if history:
        transcript_lines = []
        used = 0
        for role, text in history:
            # trim long lines so we don't blow up tokens
            t = text.strip()
            if len(t) > 400:
                t = t[:380] + " …"
            line = f"{'User' if role=='user' else 'Assistant'}: {t}"
            if used + len(line) > 1500:
                break
            transcript_lines.append(line)
            used += len(line)
        composed = (
            "Conversation so far (oldest → newest):\n"
            + "\n".join(transcript_lines)
            + "\n\nUser's new message: "
            + user_text
        )
    else:
        composed = user_text
    try:
        answer, *_ = answer_with_full_rag(composed, 5, namespace)
    except Exception:
        app.logger.exception("RAG pipeline error")
        answer = "Sorry, something went wrong. Please try again."
    try:
        send_whatsapp_text(phone_number_id, user_wa_id, answer)
    except Exception:
        app.logger.exception("Failed to send WhatsApp reply")
    try:
        log_to_csv(user_wa_id, phone_number_id, display_phone_number or "", namespace, user_text, answer)
    except Exception:
        app.logger.exception("Failed to log interaction to CSV")


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

            msg   = messages[0]
            wa_id = msg.get("from")  # user's WhatsApp number (digits)
            text  = (msg.get("text") or {}).get("body")

            if business_phone_id and wa_id and text:
                # 1) Immediate ack from the same business number
                try:
                    send_whatsapp_text(business_phone_id, wa_id, ACK_TEXT, timeout=5)
                except Exception:
                    app.logger.exception("Failed to send ack")

                # 2) Heavy work in background (keep routing info)
                executor.submit(_process_and_reply, text, wa_id, business_phone_id, display_number)

    return "EVENT_RECEIVED", 200


# Local dev only; use Gunicorn in containers
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
