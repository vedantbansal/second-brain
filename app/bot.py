import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

from app.db import (
    save_entry,
    update_entry_classification,
    update_entry_embedding,
    create_task,
)
from app.classify import classify_entry, generate_embedding
from app.query import answer_query

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Second Brain online. Send me anything — task, idea, note — "
        "and I'll save it. Or ask me a question about what you've saved."
    )


def format_classification_reply(entry_id: str, result: dict) -> str:
    """Builds the final reply text once classification completes."""
    category = result.get("category", "other")
    tags = result.get("tags") or []
    tag_str = " ".join(f"#{t}" for t in tags) if tags else ""

    label = category.capitalize()
    parts = [f"Saved as {label}"]
    if tag_str:
        parts.append(tag_str)
    if result.get("is_task") and result.get("due_date"):
        parts.append(f"due:{result['due_date']}")
    if result.get("priority"):
        parts.append(f"priority:{result['priority']}")

    return " · ".join(parts) + f"\n(id: {entry_id[:8]})"


async def classify_and_update(
    entry_id: str, text: str, chat_id: int, message_id: int, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Background task: classify via Gemini, embed, update DB, edit the original reply."""
    try:
        result = classify_entry(text)

        update_entry_classification(
            entry_id=entry_id,
            category=result.get("category", "other"),
            tags=result.get("tags", []),
            summary=result.get("summary", ""),
        )

        # Generate + store embedding for future semantic search
        try:
            embedding = generate_embedding(text)
            update_entry_embedding(entry_id, embedding)
        except Exception as embed_err:
            logger.error(f"Embedding generation failed for {entry_id}: {embed_err}")
            # Non-fatal — entry is still saved and classified, just not searchable semantically yet

        if result.get("is_task"):
            create_task(
                entry_id=entry_id,
                title=result.get("task_title") or text[:80],
                due_date=result.get("due_date"),
                priority=result.get("priority"),
            )

        reply_text = format_classification_reply(entry_id, result)

        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=reply_text
        )

    except Exception as e:
        logger.error(f"Classification failed for entry {entry_id}: {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"Saved (classification failed, raw text kept)\n(id: {entry_id[:8]})",
            )
        except Exception as edit_err:
            logger.error(f"Failed to edit message after classification error: {edit_err}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Single entry point for all text messages: detect intent (query vs capture),
    then route accordingly."""
    text = update.message.text
    user = update.effective_user

    logger.info(f"Received message from {user.id}: {text[:50]}")

    try:
        result = classify_entry(text)
    except Exception as e:
        logger.error(f"Classification/intent detection failed: {e}")
        await update.message.reply_text(
            "Something went wrong processing that — try again in a bit."
        )
        return

    if result.get("is_query"):
        # Query path: no saving, just answer via Claude tool-use
        thinking_msg = await update.message.reply_text("Looking that up...")
        try:
            answer = answer_query(text)
            await thinking_msg.edit_text(answer)
        except Exception as e:
            logger.error(f"Query answering failed: {e}")
            await thinking_msg.edit_text(
                "Had trouble answering that — try rephrasing."
            )
        return

    # Capture path: save instantly, classify already done above, just persist + embed
    try:
        saved = save_entry(raw_text=text)
        entry_id = saved.get("id")

        sent_message = await update.message.reply_text("Saved.")

        # We already have the classification result from above — finish the rest
        # (DB update, embedding, task creation, reply edit) as a background task
        context.application.create_task(
            finish_capture(
                entry_id=entry_id,
                text=text,
                result=result,
                chat_id=sent_message.chat_id,
                message_id=sent_message.message_id,
                context=context,
            )
        )

    except Exception as e:
        logger.error(f"Failed to save entry: {e}")
        await update.message.reply_text(
            "Something went wrong saving that — try again in a bit."
        )


async def finish_capture(
    entry_id: str,
    text: str,
    result: dict,
    chat_id: int,
    message_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Background: persist classification (already computed) + embedding + task, edit reply."""
    try:
        update_entry_classification(
            entry_id=entry_id,
            category=result.get("category", "other"),
            tags=result.get("tags", []),
            summary=result.get("summary", ""),
        )

        try:
            embedding = generate_embedding(text)
            update_entry_embedding(entry_id, embedding)
        except Exception as embed_err:
            logger.error(f"Embedding generation failed for {entry_id}: {embed_err}")

        if result.get("is_task"):
            create_task(
                entry_id=entry_id,
                title=result.get("task_title") or text[:80],
                due_date=result.get("due_date"),
                priority=result.get("priority"),
            )

        reply_text = format_classification_reply(entry_id, result)
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=reply_text
        )

    except Exception as e:
        logger.error(f"finish_capture failed for entry {entry_id}: {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"Saved (some processing failed)\n(id: {entry_id[:8]})",
            )
        except Exception as edit_err:
            logger.error(f"Failed to edit message after error: {edit_err}")


def build_app() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


if __name__ == "__main__":
    # Local dev only — polling mode. Production (Render) uses webhook mode via app/main.py
    application = build_app()
    logger.info("Starting bot (polling mode, local dev)...")
    application.run_polling()