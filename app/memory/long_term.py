from mem0 import MemoryClient
from app.core.config import config

mem0_client = MemoryClient(api_key=config.MEM0_API_KEY)


def save_to_ltm(user_id: str, data: dict) -> None:
    """
    Saves memory entries to Mem0 LTM.
    """

    messages = []

    if data.get("jurisdiction"):
        messages.append({
            "role": "user",
            "content": f"I am from {data['jurisdiction']}",
        })

    if data.get("last_matter_type"):
        messages.append({
            "role": "user",
            "content": (
                f"I had a {data['last_matter_type']} legal matter"
            ),
        })

    if data.get("last_user_role"):
        messages.append({
            "role": "user",
            "content": f"My role was {data['last_user_role']}",
        })

    if data.get("last_doc_path"):
        messages.append({
            "role": "user",
            "content": (
                f"I uploaded a legal document dated "
                f"{data.get('last_doc_date', 'unknown date')}"
            ),
        })

    if not messages:
        print("[MEM0] No data to save — skipping")
        return

    try:
        mem0_client.add(
            messages,
            user_id=user_id,
            output_format="v1.1",   # required by updated API
        )
        print(f"[MEM0] Saved {len(messages)} memories for {user_id}")
    except Exception as e:
        print(f"[MEM0] Save error: {e}")


def load_from_ltm(user_id: str) -> dict:
    """
    Loads memories for a user from Mem0.
    Uses filters parameter required by updated API.
    """

    try:
        # Updated API requires filters
        memories = mem0_client.get_all(
            user_id=user_id,
            output_format="v1.1",
        )

        profile = {}

        for memory in memories:
            text = memory.get("memory", "").lower()

            if "from" in text:
                profile["jurisdiction"] = memory.get("memory")
            if "matter" in text:
                profile["last_matter_type"] = memory.get("memory")
            if "role" in text:
                profile["last_user_role"] = memory.get("memory")
            if "document" in text:
                profile["last_doc_info"] = memory.get("memory")

        print(f"[MEM0] Loaded {len(memories)} memories for {user_id}")
        return profile

    except Exception as e:
        print(f"[MEM0] Load error: {e}")
        return {}


def search_ltm(user_id: str, query: str) -> list[dict]:
    """
    Searches Mem0 for relevant memories.
    """

    try:
        results = mem0_client.search(
            query,
            user_id=user_id,
            output_format="v1.1",
        )
        print(f"[MEM0] Search returned {len(results)} results")
        return results
    except Exception as e:
        print(f"[MEM0] Search error: {e}")
        return []


def clear_ltm(user_id: str) -> None:
    """
    Clears all memories for a user.
    """

    try:
        mem0_client.delete_all(user_id=user_id)
        print(f"[MEM0] Cleared memories for {user_id}")
    except Exception as e:
        print(f"[MEM0] Clear error: {e}")