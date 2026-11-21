"""User-controlled conversation memory and persistent user preferences."""

from typing import List, Dict, Any, Optional
from personal_rag.storage.sqlite_store import SQLiteStore
from personal_rag.core.types import Citation


class ConversationMemoryManager:
    """Manages short-term dialogue context and long-term explicit user memory."""

    def __init__(self, sqlite_store: SQLiteStore):
        self.sqlite_store = sqlite_store

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        citations: Optional[List[Citation]] = None
    ) -> None:
        """Store a full back-and-forth dialogue turn."""
        self.sqlite_store.save_message(session_id, "user", user_message)

        citations_payload = [
            {
                "source_path": c.source_path,
                "page_number": c.page_number,
                "excerpt": c.excerpt
            }
            for c in (citations or [])
        ]
        self.sqlite_store.save_message(
            session_id, "assistant", assistant_response, citations=citations_payload
        )

    def get_recent_dialogue(self, session_id: str, max_turns: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the last N turns of conversation history."""
        return self.sqlite_store.get_conversation_history(session_id, limit=max_turns * 2)

    def format_dialogue_context(self, session_id: str, max_turns: int = 3) -> str:
        """Format recent dialogue as context for subsequent queries."""
        history = self.get_recent_dialogue(session_id, max_turns=max_turns)
        if not history:
            return ""

        lines = ["Recent Conversation:"]
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"].strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ── Explicit Long-Term User Memory ───────────────────────────────────────

    def set_memory(self, key: str, value: str, category: str = "general") -> None:
        """Save a user-directed permanent memory or preference."""
        self.sqlite_store.set_user_memory(key, value, category)

    def get_all_memories(self) -> List[Dict[str, str]]:
        """List all saved user memories."""
        return self.sqlite_store.get_all_user_memories()

    def remove_memory(self, key: str) -> None:
        """Explicitly purge a user memory."""
        self.sqlite_store.delete_user_memory(key)

