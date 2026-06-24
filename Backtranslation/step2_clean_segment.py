"""
Step 2: Clean raw corpus text and segment into self-contained passages.
Each passage should be understandable without external context.
"""
import os
import re
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.io import read_jsonl, write_jsonl

from configs import MIN_PASSAGE_WORDS, MAX_PASSAGE_WORDS, MIN_PASSAGE_SENTENCES


def clean_text(text: str) -> str:
    """Remove markup artifacts, fix encoding, normalize whitespace."""
    # Remove leftover HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove citation brackets like [1], [2,3]
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    # Remove Wikipedia section markers like == Heading ==
    text = re.sub(r'={2,}[^=]+={2,}', '', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def split_into_paragraphs(text: str) -> list:
    """Split text on double newlines into paragraphs."""
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def count_sentences(text: str) -> int:
    """Rough sentence count by splitting on sentence-ending punctuation."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def is_self_contained(passage: str) -> bool:
    """
    Heuristic: reject passages that likely need prior context.
    - Starts with lowercase continuation words
    - Starts with pronouns referring to unknown antecedents
    - Contains "as mentioned above" / "see above"
    """
    bad_starts = [
        "however,", "therefore,", "furthermore,", "moreover,",
        "additionally,", "in addition,", "as a result,",
        "this is", "these are", "those are",
        "he ", "she ", "they ", "it ",
    ]
    lower = passage.lower().strip()
    for start in bad_starts:
        if lower.startswith(start):
            return False

    bad_phrases = ["as mentioned above", "see above", "as shown above", "the above"]
    for phrase in bad_phrases:
        if phrase in lower:
            return False

    return True


def segment_text(
    text: str,
    min_words: int = MIN_PASSAGE_WORDS,
    max_words: int = MAX_PASSAGE_WORDS,
    min_sentences: int = MIN_PASSAGE_SENTENCES,
) -> list:
    """
    Segment cleaned text into passages:
    1. Split into paragraphs
    2. Merge short paragraphs together
    3. Split overly long paragraphs at sentence boundaries
    4. Filter by min_words, min_sentences, self-contained check
    """
    paragraphs = split_into_paragraphs(text)
    passages = []
    buffer = ""

    for para in paragraphs:
        if buffer:
            combined = buffer + " " + para
        else:
            combined = para

        word_count = len(combined.split())

        if word_count < min_words:
            buffer = combined
            continue

        if word_count <= max_words:
            passages.append(combined)
            buffer = ""
        else:
            # Split at sentence boundaries
            if buffer and len(buffer.split()) >= min_words:
                passages.append(buffer)

            sentences = re.split(r'(?<=[.!?])\s+', para)
            chunk = ""
            for sent in sentences:
                if chunk:
                    test = chunk + " " + sent
                else:
                    test = sent

                if len(test.split()) > max_words and chunk:
                    passages.append(chunk)
                    chunk = sent
                else:
                    chunk = test

            if chunk and len(chunk.split()) >= min_words:
                passages.append(chunk)
            buffer = ""

    # Don't forget the buffer
    if buffer and len(buffer.split()) >= min_words:
        passages.append(buffer)

    # Final filtering
    filtered = []
    for p in passages:
        if len(p.split()) < min_words:
            continue
        if count_sentences(p) < min_sentences:
            continue
        if not is_self_contained(p):
            continue
        filtered.append(p)

    return filtered


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and segment raw corpus into passages")
    parser.add_argument("--input_path", type=str, default="corpus/raw_corpus.jsonl")
    parser.add_argument("--output_path", type=str, default="output/passages.jsonl")
    parser.add_argument("--min_words", type=int, default=MIN_PASSAGE_WORDS)
    parser.add_argument("--max_words", type=int, default=MAX_PASSAGE_WORDS)
    parser.add_argument("--min_sentences", type=int, default=MIN_PASSAGE_SENTENCES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    raw_corpus = read_jsonl(args.input_path)
    print(f"Loaded {len(raw_corpus)} raw documents")

    all_passages = []
    passage_id = 0

    for doc in raw_corpus:
        cleaned = clean_text(doc["text"])
        segments = segment_text(
            cleaned,
            min_words=args.min_words,
            max_words=args.max_words,
            min_sentences=args.min_sentences,
        )

        for seg in segments:
            all_passages.append({
                "passage_id": passage_id,
                "source": doc.get("source", "unknown"),
                "title": doc.get("title", ""),
                "text": seg,
                "word_count": len(seg.split()),
            })
            passage_id += 1

    write_jsonl(args.output_path, all_passages)
    print(f"Segmented into {len(all_passages)} passages → {args.output_path}")
    print(f"Avg words/passage: {sum(p['word_count'] for p in all_passages) / max(len(all_passages), 1):.0f}")
