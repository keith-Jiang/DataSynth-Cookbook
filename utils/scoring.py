"""LLM-as-judge quality scoring for instruction-response pairs."""
import re
import json
import asyncio
from typing import List, Dict, Any, Optional

from .api import make_chat_request, make_async_batch_chat_requests


DEFAULT_RUBRIC = """Rate the quality of the following instruction-response pair on a scale of 1-5:

1 - Very poor: The instruction is nonsensical or the response is completely irrelevant
2 - Poor: The instruction is vague/generic and the response is only loosely related
3 - Acceptable: The instruction is reasonable but the response is incomplete or partially off-topic
4 - Good: The instruction is clear and specific, the response addresses it well
5 - Excellent: The instruction is natural and precise, the response is an ideal answer

Instruction: {instruction}
Response: {response}

Output ONLY a JSON object: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""


def build_scoring_messages(
    instruction: str,
    response: str,
    rubric: str = DEFAULT_RUBRIC,
) -> List[Dict[str, str]]:
    """Build messages list for scoring a single pair."""
    prompt = rubric.format(instruction=instruction, response=response)
    return [
        {"role": "system", "content": "You are an expert data quality evaluator. Score the given instruction-response pair strictly according to the rubric."},
        {"role": "user", "content": prompt},
    ]


def parse_score_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response into {"score": int, "reason": str}. Returns score=0 on failure."""
    if not response_text:
        return {"score": 0, "reason": "empty response"}

    json_match = re.search(r'\{[^}]+\}', response_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            score = int(parsed.get("score", 0))
            reason = str(parsed.get("reason", ""))
            if 1 <= score <= 5:
                return {"score": score, "reason": reason}
        except (json.JSONDecodeError, ValueError):
            pass

    score_match = re.search(r'[Ss]core[:\s]*(\d)', response_text)
    if score_match:
        score = int(score_match.group(1))
        if 1 <= score <= 5:
            return {"score": score, "reason": response_text.strip()}

    return {"score": 0, "reason": f"parse failed: {response_text[:100]}"}


def score_pairs_sync(
    pairs: List[Dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    rubric: str = DEFAULT_RUBRIC,
    extra_body: Optional[Dict[str, Any]] = None,
    sleep_between: float = 1.0,
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Score pairs synchronously (sequential). Good for small batches.
    Each pair should have keys "instruction" and "response".
    Returns list of {"score": int, "reason": str}.
    """
    import time
    results = []
    for pair in pairs:
        messages = build_scoring_messages(pair["instruction"], pair["response"], rubric)
        resp = make_chat_request(
            messages,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=256,
            extra_body=extra_body,
        )
        content = resp["content"] if resp else ""
        results.append(parse_score_response(content))
        if sleep_between > 0:
            time.sleep(sleep_between)
    return results


async def score_pairs_async(
    pairs: List[Dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    rubric: str = DEFAULT_RUBRIC,
    extra_body: Optional[Dict[str, Any]] = None,
    max_concurrency: int = 10,
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Score pairs concurrently (async). Good for large batches.
    Returns list of {"score": int, "reason": str}.
    """
    messages_list = [
        build_scoring_messages(pair["instruction"], pair["response"], rubric)
        for pair in pairs
    ]

    responses = await make_async_batch_chat_requests(
        messages_list,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=256,
        extra_body=extra_body,
        max_concurrency=max_concurrency,
    )

    return [
        parse_score_response(r["content"] if r else "")
        for r in responses
    ]


def filter_by_score(
    records: List[Dict[str, Any]],
    min_score: int = 4,
    score_key: str = "quality_score",
) -> List[Dict[str, Any]]:
    """Keep only records where score >= min_score."""
    return [r for r in records if r.get(score_key, 0) >= min_score]
