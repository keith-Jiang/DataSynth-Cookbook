"""Backward-compatible API utils. Delegates to shared utils/api.py."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.api import (
    make_chat_request as _make_chat_request,
    make_batch_chat_requests as _make_batch_chat_requests,
)
from configs import API_KEY, BASE_URL, MODEL_NAME

EXTRA_BODY = {"thinking": {"type": "disabled"}}


def get_client():
    """Legacy: returns an OpenAI client. Kept for any direct usage."""
    from openai import OpenAI
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def make_chat_request(
    messages,
    temperature=0.7,
    max_tokens=4096,
    stop=None,
    retries=3,
):
    return _make_chat_request(
        messages,
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        extra_body=EXTRA_BODY,
        retries=retries,
    )


def make_batch_chat_requests(
    messages_list,
    temperature=0.7,
    max_tokens=4096,
    stop=None,
    sleep_between=1.0,
):
    return _make_batch_chat_requests(
        messages_list,
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        extra_body=EXTRA_BODY,
        sleep_between=sleep_between,
    )
