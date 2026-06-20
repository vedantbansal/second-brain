import os
import logging
from datetime import date
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv
from telegram import Update

from app.bot import build_app
from app.db import get_digest_tasks

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("WEBHOOK_URL") 
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BOT_CHAT_ID = os.getenv("BOT_CHAT_ID") 
DIGEST_SECRET = os.getenv("DIGEST_SECRET")

telegram_app = build_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize bot, set webhook
    await telegram_app.initialize()
    await telegram_app.start()

    if WEBHOOK_URL:
        webhook_path = f"{WEBHOOK_URL}/webhook/{TELEGRAM_BOT_TOKEN}"
        success = await telegram_app.bot.set_webhook(url=webhook_path)
        if success:
            logger.info(f"Webhook set successfully: {webhook_path}")
        else:
            logger.error(f"set_webhook returned False for: {webhook_path}")

        info = await telegram_app.bot.get_webhook_info()
        logger.info(f"Telegram-confirmed webhook info: {info.to_dict()}")
    else:
        logger.warning("WEBHOOK_URL not set — webhook not configured")

    yield

    await telegram_app.bot.delete_webhook()
    await telegram_app.stop()
    await telegram_app.shutdown()


app = FastAPI(lifespan=lifespan)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    """Telegram pushes updates here instead of us polling for them."""
    if token != TELEGRAM_BOT_TOKEN:
        return Response(status_code=403)

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)

    return Response(status_code=200)


def format_digest(tasks: list[dict]) -> str:
    """Builds the morning digest message text."""
    if not tasks:
        return "Good morning. No open tasks due today or overdue. Clear runway."

    today_str = date.today().isoformat()
    overdue = [t for t in tasks if t["due_date"] < today_str]
    due_today = [t for t in tasks if t["due_date"] == today_str]

    lines = ["Good morning. Here's today:"]

    if overdue:
        lines.append("\nOverdue:")
        for t in overdue:
            priority_tag = f" ({t['priority']})" if t.get("priority") else ""
            lines.append(f"  • {t['title']}{priority_tag} — was due {t['due_date']}")

    if due_today:
        lines.append("\nDue today:")
        for t in due_today:
            priority_tag = f" ({t['priority']})" if t.get("priority") else ""
            lines.append(f"  • {t['title']}{priority_tag}")

    return "\n".join(lines)


@app.post("/trigger-digest")
async def trigger_digest(request: Request):
    secret = request.headers.get("X-Digest-Secret") or request.query_params.get("secret")

    if not DIGEST_SECRET or secret != DIGEST_SECRET:
        return Response(status_code=403)

    if not BOT_CHAT_ID:
        logger.error("MY_CHAT_ID not set — cannot send digest")
        return Response(status_code=500)

    try:
        tasks = get_digest_tasks(today=date.today().isoformat())
        message = format_digest(tasks)
        await telegram_app.bot.send_message(chat_id=int(BOT_CHAT_ID), text=message)
        logger.info(f"Digest sent: {len(tasks)} tasks")
        return {"status": "sent", "task_count": len(tasks)}
    except Exception as e:
        logger.error(f"Digest send failed: {e}")
        return Response(status_code=500)