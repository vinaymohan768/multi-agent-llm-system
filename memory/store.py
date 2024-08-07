"""
memory/store.py

PostgreSQL-backed conversation memory with rolling summarization.

Every message is stored in full. Once the history crosses SUMMARY_THRESHOLD,
the older messages get summarized by the LLM and that summary is stored in
memory_summaries. On the next turn, the summary is prepended as a system
message so the agent still has context without blowing up the token count.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

import psycopg2
import psycopg2.extras
from openai import OpenAI

log = logging.getLogger(__name__)

SUMMARY_THRESHOLD = int(os.getenv("SUMMARY_THRESHOLD", "20"))  # messages before summarizing
RECENT_MESSAGE_COUNT = int(os.getenv("RECENT_MESSAGE_COUNT", "10"))  # keep last N verbatim

DB_DSN = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'agentdb')} "
    f"user={os.getenv('POSTGRES_USER', 'agent')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'agent')}"
)


@dataclass
class Message:
    role: str
    content: str
    tool_name: Optional[str] = None


class ConversationMemory:
    """Per-session conversation memory backed by PostgreSQL."""

    def __init__(self, session_id: str, openai_client: OpenAI, db_conn=None):
        self.session_id = session_id
        self.client = openai_client
        self._conn = db_conn

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(DB_DSN)
            self._conn.autocommit = True
        return self._conn

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, role: str, content: str, tool_name: Optional[str] = None):
        """Append a message to the conversation history."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_memory (session_id, role, content, tool_name)
                VALUES (%s, %s, %s, %s)
                """,
                (self.session_id, role, content, tool_name),
            )
        log.debug("Memory add | session=%s role=%s", self.session_id, role)

        # Trigger summarization if history is getting long
        if self._message_count() > SUMMARY_THRESHOLD:
            self._summarize()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_context(self) -> list[dict]:
        """Return [optional summary system message] + last N messages verbatim."""
        messages = []

        summary = self._get_summary()
        if summary:
            messages.append({
                "role": "system",
                "content": f"[Conversation so far]\n{summary}",
            })

        recent = self._get_recent_messages(RECENT_MESSAGE_COUNT)
        messages.extend(recent)

        return messages

    def get_full_history(self) -> list[dict]:
        """Returns the complete message history (for export/debugging)."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content, tool_name, created_at
                FROM conversation_memory
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (self.session_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def clear(self):
        """Delete all messages and summaries for this session."""
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversation_memory WHERE session_id = %s",
                (self.session_id,),
            )
            cur.execute(
                "DELETE FROM memory_summaries WHERE session_id = %s",
                (self.session_id,),
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _message_count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM conversation_memory WHERE session_id = %s",
                (self.session_id,),
            )
            return cur.fetchone()[0]

    def _get_recent_messages(self, n: int) -> list[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM conversation_memory
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub
                ORDER BY created_at ASC
                """,
                (self.session_id, n),
            )
            return [{"role": r["role"], "content": r["content"]} for r in cur.fetchall()]

    def _get_summary(self) -> Optional[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT summary FROM memory_summaries WHERE session_id = %s",
                (self.session_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def _summarize(self):
        """Summarize old messages and upsert the result into memory_summaries."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM conversation_memory
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (self.session_id,),
            )
            all_messages = [dict(r) for r in cur.fetchall()]

        # Keep recent messages verbatim; summarize the rest
        to_summarize = all_messages[:-RECENT_MESSAGE_COUNT]
        if not to_summarize:
            return

        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content'][:500]}" for m in to_summarize
        )

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following conversation concisely, preserving "
                            "key facts, decisions, and context that will be needed to "
                            "continue the conversation coherently. Be specific."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
                temperature=0,
                max_tokens=500,
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            log.warning("Summarization failed: %s", e)
            return

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_summaries (session_id, summary, message_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE
                    SET summary = EXCLUDED.summary,
                        message_count = EXCLUDED.message_count,
                        updated_at = NOW()
                """,
                (self.session_id, summary, len(to_summarize)),
            )
        log.info("Memory summarized | session=%s covered=%d messages", self.session_id, len(to_summarize))

