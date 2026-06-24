"""Configurable text filtering utilities."""
import re
from typing import List, Tuple, Callable, Dict, Any, Optional


def word_count_filter(text: str, min_words: int = 5, max_words: int = 500) -> bool:
    """Return True if text passes word count bounds."""
    count = len(text.split())
    return min_words <= count <= max_words


def blocked_words_filter(
    text: str,
    blocked_words: Optional[List[str]] = None,
) -> bool:
    """Return True if text contains none of the blocked words (case-insensitive word boundary match)."""
    if not blocked_words:
        return True
    for word in blocked_words:
        if re.compile(r"\b({0})\b".format(re.escape(word)), flags=re.IGNORECASE).search(text):
            return False
    return True


def starts_with_valid_char(text: str) -> bool:
    """Return True if text starts with an alphanumeric character."""
    if not text:
        return False
    return text[0].isalnum()


def filter_invalid_instances(instances: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Remove instances where:
    - input == output
    - output is empty
    - input or output ends with ":"
    Each instance is (instruction, input, output).
    """
    filtered = []
    for inst in instances:
        instruction, inp, out = inst
        if inp == out:
            continue
        if out == "":
            continue
        if inp.strip().endswith(":") or out.strip().endswith(":"):
            continue
        filtered.append(inst)
    return filtered


def filter_duplicate_instances(instances: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Remove instances with conflicting outputs for same input,
    and deduplicate exact (input, output) pairs.
    """
    same_input_diff_output = False
    for i in range(1, len(instances)):
        for j in range(0, i):
            if instances[i][1] == "":
                continue
            if instances[i][1] == instances[j][1] and instances[i][2] != instances[j][2]:
                same_input_diff_output = True
                break
    if same_input_diff_output:
        return []

    seen = set()
    unique = []
    for inst in instances:
        key = (inst[1], inst[2])
        if key not in seen:
            seen.add(key)
            unique.append(inst)
    return unique


def apply_filters(
    records: List[Dict[str, Any]],
    filters: List[Callable[[Dict[str, Any]], bool]],
) -> List[Dict[str, Any]]:
    """Apply a chain of filter functions. Keep records where ALL filters return True."""
    result = []
    for record in records:
        if all(f(record) for f in filters):
            result.append(record)
    return result
