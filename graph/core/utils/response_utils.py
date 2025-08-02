import re
from typing import Optional


def extract_answer(response: str) -> str:
    """Extract content between <answer> tags."""
    start_tag = "<answer>"
    end_tag = "</answer>"
    
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)
    
    if start_idx != -1 and end_idx != -1:
        return response[start_idx + len(start_tag):end_idx].strip()
    
    return response.strip()
