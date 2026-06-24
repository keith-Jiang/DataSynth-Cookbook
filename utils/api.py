"""Sync and async OpenAI-compatible API wrappers with retry and rate limiting."""
import time
import asyncio
from typing import List, Dict, Any, Optional

from openai import OpenAI, AsyncOpenAI


def make_chat_request(
    messages: List[Dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stop: Optional[List[str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    initial_backoff: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    Single synchronous chat completion with exponential backoff retry.
    Returns {"content": str, "finish_reason": str, "usage": {...}} or None.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    backoff = initial_backoff

    for attempt in range(retries + 1):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if stop:
                kwargs["stop"] = stop
            if extra_body:
                kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**kwargs)
            return {
                "content": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            }
        except Exception as e:
            if attempt < retries:
                print(f"[API Error] {e} | Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff *= 1.5
            else:
                print(f"[API Error] {e} | All {retries} retries exhausted.")
    return None


def make_batch_chat_requests(
    messages_list: List[List[Dict[str, str]]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stop: Optional[List[str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    sleep_between: float = 1.0,
) -> List[Optional[Dict[str, Any]]]:
    """Sequential batch requests with configurable sleep between calls."""
    results = []
    for messages in messages_list:
        result = make_chat_request(
            messages,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
        )
        results.append(result)
        if sleep_between > 0:
            time.sleep(sleep_between)
    return results


async def _async_single_request(
    messages: List[Dict[str, str]],
    *,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    stop: Optional[List[str]],
    extra_body: Optional[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
    retries: int = 3,
    initial_backoff: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Single async request with semaphore-based concurrency control."""
    async with semaphore:
        backoff = initial_backoff
        for attempt in range(retries + 1):
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if stop:
                    kwargs["stop"] = stop
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = await client.chat.completions.create(**kwargs)
                return {
                    "content": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                }
            except Exception as e:
                if attempt < retries:
                    print(f"[Async API Error] {e} | Retrying in {backoff:.0f}s...")
                    await asyncio.sleep(backoff)
                    backoff *= 1.5
                else:
                    print(f"[Async API Error] {e} | All {retries} retries exhausted.")
        return None


async def make_async_batch_chat_requests(
    messages_list: List[List[Dict[str, str]]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stop: Optional[List[str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    max_concurrency: int = 10,
    retries: int = 3,
) -> List[Optional[Dict[str, Any]]]:
    """
    Async batch: fires all requests concurrently, limited by semaphore.
    Order of results matches order of input messages_list.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _async_single_request(
            messages,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
            semaphore=semaphore,
            retries=retries,
        )
        for messages in messages_list
    ]
    results = await asyncio.gather(*tasks)
    await client.close()
    return list(results)


def run_async_batch(
    messages_list: List[List[Dict[str, str]]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stop: Optional[List[str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    max_concurrency: int = 10,
) -> List[Optional[Dict[str, Any]]]:
    """
    Convenience wrapper: runs async batch from synchronous code.
    Handles event loop creation / nest_asyncio if needed.
    """
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
    except RuntimeError:
        pass

    return asyncio.run(
        make_async_batch_chat_requests(
            messages_list,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
            max_concurrency=max_concurrency,
        )
    )
