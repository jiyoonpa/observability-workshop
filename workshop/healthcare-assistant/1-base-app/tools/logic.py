"""
Call-summary domain tools.

- get_call_transcript: fixed parameterized lookup of a call + transcript
- get_customer_profile: fixed parameterized lookup of a customer
- search_policy_guide: semantic search against carrier policy + summary guide
- send_summary: mock SMS send of a summary (no real delivery)
"""
import json
import logging
from typing import Optional, Tuple

from langchain_postgres import PGVector

from config import DOMAIN, load_config
from helpers.pgvector_utils import get_pgvector_store
from helpers.sql_utils import execute_sql, relational_table_name
from rag import get_rag_system

_CALL_TABLE = relational_table_name(DOMAIN, "call")
_CUSTOMER_TABLE = relational_table_name(DOMAIN, "customer")

_vector_store: Optional[PGVector] = None
_embedding_model: Optional[str] = None
_collection_name_cached: Optional[str] = None


def _get_vector_store() -> Tuple[PGVector, str]:
    global _vector_store, _embedding_model, _collection_name_cached

    app_config = load_config()
    embedding_model = (
        app_config.get("vectorstore", {}).get("embedding_model") or "text-embedding-3-large"
    )

    if (
        _vector_store is not None
        and _collection_name_cached is not None
        and _embedding_model == embedding_model
    ):
        return _vector_store, _collection_name_cached

    _vector_store, collection_name = get_pgvector_store(DOMAIN, embedding_model)
    _embedding_model = embedding_model
    _collection_name_cached = collection_name
    return _vector_store, collection_name


def _dumps(payload) -> str:
    # ensure_ascii=False keeps Korean readable for the LLM and in traces.
    return json.dumps(payload, ensure_ascii=False, default=str)


async def get_call_transcript(call_id: str) -> str:
    """Retrieve a call record with its transcript by call ID."""
    call_id = (call_id or "").strip().upper()
    if not call_id:
        return _dumps({"error": "call_id is required"})

    try:
        result = execute_sql(
            "SELECT call_id, customer_id, call_datetime, duration_sec, agent_name, "
            f'category, transcript FROM "{_CALL_TABLE}" WHERE call_id = :call_id',
            {"call_id": call_id},
        )
    except Exception as e:
        return _dumps({"error": str(e), "call_id": call_id})

    if not result["rows"]:
        return _dumps({"error": f"call not found: {call_id}"})
    return _dumps(result["rows"][0])


async def get_customer_profile(customer_id: str) -> str:
    """Retrieve a customer profile by customer ID."""
    customer_id = (customer_id or "").strip().upper()
    if not customer_id:
        return _dumps({"error": "customer_id is required"})

    try:
        result = execute_sql(
            "SELECT customer_id, name, phone, address, plan, join_date "
            f'FROM "{_CUSTOMER_TABLE}" WHERE customer_id = :customer_id',
            {"customer_id": customer_id},
        )
    except Exception as e:
        return _dumps({"error": str(e), "customer_id": customer_id})

    if not result["rows"]:
        return _dumps({"error": f"customer not found: {customer_id}"})
    return _dumps(result["rows"][0])


async def search_policy_guide(query: str) -> str:
    """Search carrier policy and summary guide using semantic vector search."""
    q = query
    try:
        _get_vector_store()
    except Exception as e:
        return _dumps({"error": str(e), "query": q})

    try:
        rag_system = get_rag_system(top_k=3)
        raw = await rag_system.search(q)
    except Exception as e:
        logging.exception("search_policy_guide search failed")
        return _dumps({"error": str(e), "query": q})

    return _dumps([raw])


async def send_summary(call_id: str, summary: str, to_phone: Optional[str] = None) -> str:
    """Mock-send a call summary by SMS. No real message is delivered."""
    call_id = (call_id or "").strip().upper()
    if not call_id:
        return _dumps({"error": "call_id is required"})

    try:
        rows = execute_sql(
            f'SELECT c.call_id, u.phone FROM "{_CALL_TABLE}" c '
            f'JOIN "{_CUSTOMER_TABLE}" u ON u.customer_id = c.customer_id '
            "WHERE c.call_id = :call_id",
            {"call_id": call_id},
        )["rows"]
    except Exception as e:
        return _dumps({"error": str(e), "call_id": call_id})

    if not rows:
        return _dumps({"error": f"call not found: {call_id}"})

    recipient = (to_phone or "").strip() or rows[0]["phone"]
    return _dumps(
        {"status": "sent (mock)", "call_id": call_id, "to": recipient, "summary": summary}
    )


TOOLS = [get_call_transcript, get_customer_profile, search_policy_guide, send_summary]
