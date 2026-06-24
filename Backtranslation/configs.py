import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL_NAME = "deepseek-v4-flash"
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
EXTRA_BODY = {"thinking": {"type": "disabled"}}

# Step 1: Corpus collection
CORPUS_SOURCES = {
    "wikipedia": {
        "lang": "en",
        "max_articles": 500,
        "min_article_length": 200,
    },
    "stackexchange": {
        "site": "stackoverflow",
        "max_posts": 300,
        "min_score": 5,
        "tagged": "python",
    },
}
SCRAPE_DELAY = 1.5

# Step 2: Segmentation
MIN_PASSAGE_WORDS = 50
MAX_PASSAGE_WORDS = 500
MIN_PASSAGE_SENTENCES = 3

# Step 3: Backtranslation
BACKTRANSLATE_TEMPERATURE = 0.7
BACKTRANSLATE_MAX_TOKENS = 256
ASYNC_MAX_CONCURRENCY = 10

# Step 4: Scoring & filtering
SCORE_TEMPERATURE = 0.1
SCORE_MAX_TOKENS = 256
MIN_QUALITY_SCORE = 4

# Step 5: Response rewriting
REWRITE_TEMPERATURE = 0.5
REWRITE_MAX_TOKENS = 2048
REWRITE_SCORE_RANGE = (4, 4)  # only rewrite score=4 pairs
