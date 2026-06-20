import os
import json
import logging
import time
import random
from datetime import date
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"  # highest free-tier RPM, fine for classification

CLASSIFICATION_PROMPT = """You are the routing brain for a personal "second brain" note-taking system.
Given an incoming message, first decide: is this a QUERY (the user is asking a question
about their stored notes/tasks, e.g. "what tasks do I have today", "find my notes about X",
"when did I save that idea about Y") OR is this a NEW ENTRY to capture (a task, idea, note,
journal entry, etc. that should be saved)?

Return ONLY a JSON object (no markdown, no preamble) with this shape:

{{
  "is_query": true | false,
  "category": "task" | "idea" | "work" | "study" | "journal" | "reference" | "other",
  "tags": ["lowercase", "free-form", "tags"],
  "summary": "a short one-line summary of the entry, max 10 words",
  "is_task": true | false,
  "task_title": "short actionable title, only if is_task is true, else null",
  "due_date": "YYYY-MM-DD or null if no date mentioned or inferable",
  "priority": "low" | "medium" | "high" | null
}}

If is_query is true, the category/tags/summary/is_task/due_date/priority fields don't matter —
just set is_query: true and fill the rest with reasonable defaults (category: "other", is_task: false).

Today's date is {today}. Resolve relative dates ("tomorrow", "next monday") into actual YYYY-MM-DD dates.

Message: "{text}"
"""

EMBEDDING_MODEL = "models/text-embedding-004"

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 5  # free tier RPM is tight; give it real room to recover


def classify_entry(text: str) -> dict:
    """Calls Gemini to classify a raw entry. Returns parsed dict.
    Retries with exponential backoff on rate-limit (429) errors.
    Raises on final failure — caller should handle gracefully."""
    prompt = CLASSIFICATION_PROMPT.format(today=date.today().isoformat(), text=text)
    model = genai.GenerativeModel(MODEL_NAME)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {raw[:200]}")
                raise ValueError(f"Invalid classification response: {e}")

        except ResourceExhausted as e:
            last_error = e
            delay = BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                f"Rate limited (attempt {attempt + 1}/{MAX_RETRIES}), "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)

    raise RuntimeError(f"Gemini classification failed after {MAX_RETRIES} retries: {last_error}")


def generate_embedding(text: str) -> list[float]:
    """Generates a 768-dim embedding vector for the given text using Gemini.
    Used for semantic search storage and query-time comparison."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def generate_query_embedding(text: str) -> list[float]:
    """Generates an embedding for a search QUERY (not a stored document).
    Gemini's embedding model distinguishes query vs document intent for better retrieval."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]