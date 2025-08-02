from typing import List, Dict, Any


def beautify_chat_history(history: List[Dict[str, Any]]) -> str:
    """Convert chat history to readable string."""
    return "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in history
    )


def get_last_human_message(messages: List[Dict[str, Any]]) -> str:
    """Get the last user message."""
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    if not user_messages:
        raise ValueError("No user messages found in the message history")
    return user_messages[-1]["content"]
