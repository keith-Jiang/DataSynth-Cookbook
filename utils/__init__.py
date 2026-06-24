"""data_synthesis shared utilities."""

from .io import read_jsonl, write_jsonl, append_jsonl, iter_jsonl
from .api import (
    make_chat_request,
    make_batch_chat_requests,
    make_async_batch_chat_requests,
    run_async_batch,
)
from .dedup import (
    create_rouge_scorer,
    compute_rouge_scores,
    is_duplicate,
    deduplicate_by_rouge,
    deduplicate_exact,
)
from .filtering import (
    word_count_filter,
    blocked_words_filter,
    starts_with_valid_char,
    filter_invalid_instances,
    filter_duplicate_instances,
    apply_filters,
)
from .scoring import (
    score_pairs_sync,
    score_pairs_async,
    filter_by_score,
    build_scoring_messages,
    parse_score_response,
)
