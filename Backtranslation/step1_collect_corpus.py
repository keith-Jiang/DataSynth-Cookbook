"""
Step 1: Collect high-quality text corpus from Wikipedia and StackExchange.
Uses public APIs — no authentication needed, respects rate limits.
"""
import os
import sys
import time
import argparse
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.io import read_jsonl, append_jsonl

from configs import CORPUS_SOURCES, SCRAPE_DELAY


def fetch_wikipedia_articles(
    max_articles: int = 500,
    lang: str = "en",
    min_length: int = 200,
    output_path: str = "corpus/raw_corpus.jsonl",
) -> int:
    """
    Fetch random Wikipedia articles via the MediaWiki API.
    Uses action=query with generator=random for diverse topics.
    """
    base_url = f"https://{lang}.wikipedia.org/w/api.php"
    collected = 0
    existing_titles = set()

    existing = read_jsonl(output_path)
    for r in existing:
        if r.get("source") == "wikipedia":
            existing_titles.add(r["title"])
            collected += 1

    print(f"[Wikipedia] Already have {collected} articles, targeting {max_articles}")

    while collected < max_articles:
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 10,
            "prop": "extracts",
            "explaintext": True,
            "exlimit": "max",
        }

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[Wikipedia] Request failed: {e}, retrying...")
            time.sleep(SCRAPE_DELAY * 3)
            continue

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if collected >= max_articles:
                break

            title = page.get("title", "")
            text = page.get("extract", "")

            if title in existing_titles:
                continue
            if len(text.split()) < min_length:
                continue

            append_jsonl(output_path, {
                "source": "wikipedia",
                "title": title,
                "text": text,
            })
            existing_titles.add(title)
            collected += 1

        print(f"[Wikipedia] Collected {collected}/{max_articles}")
        time.sleep(SCRAPE_DELAY)

    return collected


def fetch_stackoverflow_posts(
    max_posts: int = 300,
    site: str = "stackoverflow",
    min_score: int = 5,
    tagged: str = "python",
    output_path: str = "corpus/raw_corpus.jsonl",
) -> int:
    """
    Fetch high-quality StackOverflow answers via the StackExchange API.
    No authentication needed for low-volume usage (300 req/day with key, 30/min without).
    """
    base_url = f"https://api.stackexchange.com/2.3/questions"
    collected = 0
    existing_ids = set()

    existing = read_jsonl(output_path)
    for r in existing:
        if r.get("source") == "stackexchange":
            existing_ids.add(r.get("post_id"))
            collected += 1

    print(f"[StackExchange] Already have {collected} posts, targeting {max_posts}")

    page = 1
    while collected < max_posts:
        params = {
            "order": "desc",
            "sort": "votes",
            "tagged": tagged,
            "site": site,
            "filter": "withbody",
            "pagesize": 20,
            "page": page,
        }

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[StackExchange] Request failed: {e}, retrying...")
            time.sleep(SCRAPE_DELAY * 3)
            continue

        items = data.get("items", [])
        if not items:
            print("[StackExchange] No more items, stopping.")
            break

        for item in items:
            if collected >= max_posts:
                break

            post_id = item.get("question_id")
            if post_id in existing_ids:
                continue

            score = item.get("score", 0)
            if score < min_score:
                continue

            title = item.get("title", "")
            body = item.get("body", "")

            # Strip HTML tags (rough, since API returns HTML body)
            import re
            clean_body = re.sub(r'<[^>]+>', '', body)
            clean_body = clean_body.strip()

            if len(clean_body.split()) < 30:
                continue

            append_jsonl(output_path, {
                "source": "stackexchange",
                "title": title,
                "text": clean_body,
                "post_id": post_id,
                "score": score,
                "tags": item.get("tags", []),
            })
            existing_ids.add(post_id)
            collected += 1

        print(f"[StackExchange] Collected {collected}/{max_posts}")
        page += 1

        if data.get("has_more") is False:
            break

        # StackExchange rate limit: respect backoff
        backoff = data.get("backoff", SCRAPE_DELAY)
        time.sleep(max(backoff, SCRAPE_DELAY))

    return collected


def parse_args():
    parser = argparse.ArgumentParser(description="Collect corpus from web sources")
    parser.add_argument("--output_dir", type=str, default="corpus")
    parser.add_argument("--max_wiki", type=int, default=None)
    parser.add_argument("--max_stack", type=int, default=None)
    parser.add_argument("--skip_wiki", action="store_true")
    parser.add_argument("--skip_stack", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "raw_corpus.jsonl")

    wiki_cfg = CORPUS_SOURCES["wikipedia"]
    stack_cfg = CORPUS_SOURCES["stackexchange"]

    total = 0

    if not args.skip_wiki:
        max_wiki = args.max_wiki or wiki_cfg["max_articles"]
        count = fetch_wikipedia_articles(
            max_articles=max_wiki,
            lang=wiki_cfg["lang"],
            min_length=wiki_cfg["min_article_length"],
            output_path=output_path,
        )
        total += count
        print(f"[Wikipedia] Done: {count} articles")

    if not args.skip_stack:
        max_stack = args.max_stack or stack_cfg["max_posts"]
        count = fetch_stackoverflow_posts(
            max_posts=max_stack,
            site=stack_cfg["site"],
            min_score=stack_cfg["min_score"],
            tagged=stack_cfg["tagged"],
            output_path=output_path,
        )
        total += count
        print(f"[StackExchange] Done: {count} posts")

    print(f"\nTotal corpus size: {total} documents → {output_path}")
