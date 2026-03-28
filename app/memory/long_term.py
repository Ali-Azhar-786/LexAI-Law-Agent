from mem0 import MemoryClient
from app.core.config import config

mem0_client = MemoryClient(api_key=config.MEM0_API_KEY)


def save_to_ltm(user_id: str, data: dict) -> None:
    if not mem0_client:
        return

    messages = []

    if data.get("jurisdiction") not in (None, "", "unspecified"):
        messages.append({
            "role": "user",
            "content": f"I am from {data['jurisdiction']}",
        })

    if data.get("last_matter_type") not in (None, "", "unspecified"):
        messages.append({
            "role": "user",
            "content": f"I had a {data['last_matter_type']} legal matter",
        })

    if data.get("last_user_role") not in (None, "", "unspecified"):
        messages.append({
            "role": "user",
            "content": f"My role was {data['last_user_role']}",
        })

    if not messages:
        return

    try:
        # v1.0.7 requires keyword argument
        mem0_client.add(messages=messages, user_id=user_id)
        print(f"[MEM0] Saved {len(messages)} memories for {user_id}")
    except Exception as e:
        print(f"[MEM0] Save error (non-fatal): {e}")


def load_from_ltm(user_id: str) -> dict:
    if not mem0_client:
        return {}

    try:
        memories = mem0_client.get_all(user_id=user_id)

        if not memories:
            return {}

        # v1.0.7 returns list directly
        if isinstance(memories, dict):
            memories = memories.get("results", [])

        profile = {}
        for memory in memories:
            text = memory.get("memory", "").lower()
            if "from" in text:
                profile["jurisdiction"] = memory.get("memory")
            if "matter" in text:
                profile["last_matter_type"] = memory.get("memory")
            if "role" in text:
                profile["last_user_role"] = memory.get("memory")

        print(f"[MEM0] Loaded {len(memories)} memories for {user_id}")
        return profile

    except Exception as e:
        print(f"[MEM0] Load error (non-fatal): {e}")
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