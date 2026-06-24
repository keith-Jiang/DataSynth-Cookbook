import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
if not os.environ.get("DEEPSEEK_API_KEY"):
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL_NAME = "deepseek-v4-flash"
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"

# Step 1: Instruction generation
NUM_PROMPT_INSTRUCTIONS = 8
NUM_INSTRUCTIONS_TO_GENERATE = 100
ROUGE_THRESHOLD = 0.7

# Step 2: Evol-Instruct
EVOLVE_RATIO = 0.5
EVOLVE_IN_DEPTH_RATIO = 0.6
EVOLVE_BREADTH_COUNT = 2
MAX_EVOLVED_INSTRUCTIONS = 200

# Step 3: Instance generation with personas
PERSONA_TEMPERATURE = 0.3
MAX_INSTANCES_PER_TASK = 5

# API parameters
REQUEST_BATCH_SIZE = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
