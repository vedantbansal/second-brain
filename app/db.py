import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def save_entry(raw_text: str, source: str = "telegram") -> dict:
    """Insert a raw captured entry into the entries table."""
    response = (
        supabase.table("entries")
        .insert({"raw_text": raw_text, "source": source})
        .execute()
    )
    return response.data[0] if response.data else {}


def update_entry_classification(
    entry_id: str, category: str, tags: list, summary: str
) -> dict:
    """Update an entry with classification results."""
    response = (
        supabase.table("entries")
        .update(
            {
                "category": category,
                "tags": tags,
                "summary": summary,
                "classified_at": "now()",
            }
        )
        .eq("id", entry_id)
        .execute()
    )
    return response.data[0] if response.data else {}


def update_entry_embedding(entry_id: str, embedding: list) -> None:
    """Store the embedding vector for an entry."""
    supabase.table("entries").update({"embedding": embedding}).eq(
        "id", entry_id
    ).execute()


def create_task(
    entry_id: str, title: str, due_date: str | None, priority: str | None
) -> dict:
    """Insert a row into the dedicated tasks table, linked to its source entry."""
    response = (
        supabase.table("tasks")
        .insert(
            {
                "entry_id": entry_id,
                "title": title,
                "due_date": due_date,
                "priority": priority,
            }
        )
        .execute()
    )
    return response.data[0] if response.data else {}


def query_tasks(
    status: str | None = None,
    due_before: str | None = None,
    due_after: str | None = None,
) -> list[dict]:
    """Structured filter query against the tasks table."""
    query = supabase.table("tasks").select("*")
    if status:
        query = query.eq("status", status)
    if due_before:
        query = query.lte("due_date", due_before)
    if due_after:
        query = query.gte("due_date", due_after)
    response = query.order("due_date", desc=False).execute()
    return response.data or []

def get_digest_tasks(today: str) -> list[dict]:
    """Tasks due today or overdue, still open. Used by the morning digest."""
    response = (
        supabase.table("tasks")
        .select("*")
        .eq("status", "open")
        .lte("due_date", today)
        .order("due_date", desc=False)
        .execute()
    )
    return response.data or []


def semantic_search_entries(
    query_embedding: list, match_count: int = 5, match_threshold: float = 0.5
) -> list[dict]:
    """Calls the match_entries RPC for vector similarity search."""
    response = supabase.rpc(
        "match_entries",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()
    return response.data or []