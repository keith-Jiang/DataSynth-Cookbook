# Data Synthesis

实用数据合成方法的工程实现合集。每条 pipeline 独立可运行，共享 `utils/` 工具库。

## 项目结构

```
DataSynth-Cookbook/
├── utils/                              # 共享工具库（API调用、去重、过滤、质量打分）
├── Instruct-data/                      # 指令数据合成方法集合
│   ├── self-instruct/                  #   Self-Instruct + Evol-Instruct + Persona
│   └── backtranslation/                #   指令回译 (Humpback + Back-and-Forth)
├── cot-data/                           # Chain-of-Thought 推理数据合成
├── agent-data/                         # Agent trajectory 数据合成
├── knowledge-graph-data/               # 知识图谱数据合成
├── merge_datasets.py                   # 跨 pipeline 合并去重
└── requirements.txt
```

## 方法概览

| 目录 | 方法 | 核心思路 | 论文来源 |
|------|------|----------|----------|
| `Instruct-data/self-instruct/` | Self-Instruct + Evol-Instruct + Persona | 从种子指令出发，few-shot 生成 → 进化 → 多角色实例生成 | Self-Instruct (2022), WizardLM (2023) |
| `Instruct-data/backtranslation/` | Backtranslation | 从高质量文本出发，反向生成指令 → 打分过滤 → 改写提质 | Humpback (2023), Back-and-Forth (2024) |
| `cot-data/` | TBD | Chain-of-Thought 推理链数据合成 | — |
| `agent-data/` | TBD | Agent 交互轨迹数据合成 | — |
| `knowledge-graph-data/` | TBD | 知识图谱结构化数据合成 | — |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API key
echo "DEEPSEEK_API_KEY=your_key_here" > .env
```

### 运行 Self-Instruct

```bash
cd Instruct-data/self-instruct
python run_pipeline.py --num_instructions 20 --evolve_ratio 0.3
```

### 运行 Backtranslation

```bash
cd Instruct-data/backtranslation

# 小规模测试
python run_pipeline.py --max_wiki 20 --max_stack 10 --sync

# 正常跑（async，快 5-10x）
python run_pipeline.py --max_wiki 100 --max_stack 50
```

### 合并数据

```bash
python merge_datasets.py --skip_missing
# 加 ROUGE 去重（慢但更彻底）
python merge_datasets.py --rouge_dedup --rouge_threshold 0.7
```

## 共享工具库 utils/

所有 pipeline 共享的基础能力：

| 模块 | 功能 |
|------|------|
| `utils/api.py` | sync + async OpenAI API 调用，支持重试退避、并发控制 |
| `utils/dedup.py` | ROUGE-L 语义去重 + 精确去重 |
| `utils/filtering.py` | 可组合的文本过滤链 |
| `utils/scoring.py` | LLM-as-judge 质量打分（5分制） |
| `utils/io.py` | JSONL 读写工具 |

### 在自己的脚本中使用

```python
import sys; sys.path.insert(0, "/path/to/data_synthesis")

from utils import run_async_batch, is_duplicate, create_rouge_scorer

# Async 并发调用
results = run_async_batch(
    messages_list=[...],
    model="deepseek-v4-flash",
    api_key="...",
    base_url="https://api.deepseek.com",
    max_concurrency=10,
)

# ROUGE 去重
scorer = create_rouge_scorer()
duplicate = is_duplicate("new instruction", existing_pool, scorer, threshold=0.7)
```

## Backtranslation 方法详解

指令回译的核心思路：**不从指令出发造数据，而是从已有高质量文本出发，反向推导指令**。

```
高质量文本 (Wikipedia, StackOverflow, ...)
    │
    ▼  Step 1: 语料收集
    │   Wikipedia API + StackExchange API → raw_corpus.jsonl
    ▼
    │  Step 2: 清洗分段
    │   去标记 / 分段 / 自包含检查 → passages.jsonl
    ▼
    │  Step 3: 指令回译
    │   LLM: "什么指令会产出这段文本？" → backtranslated_pairs.jsonl
    ▼
    │  Step 4: 质量打分
    │   LLM-as-judge 5分制评估 → scored_pairs.jsonl / high_quality_pairs.jsonl
    ▼
    │  Step 5: 回答改写 (可选)
    │   对 score=4 的 pair 改写 response 提质 → final_backtranslation_data.jsonl
    ▼
最终训练数据
```

**为什么好用**：
- 生成的数据天然更"真实"（回答来自真实文本，非 LLM 编造）
- 不需要种子指令，可利用任意领域的高质量文本
- 质量打分自动过滤噪声，数据越多打分越准（迭代改进）

## 参考论文

- [Self-Instruct](https://arxiv.org/abs/2212.10560) — Wang et al., 2022
- [Evol-Instruct / WizardLM](https://arxiv.org/abs/2304.12244) — Xu et al., 2023
- [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259) — Li et al. (Meta), 2023
- [Better Alignment with Instruction Back-and-Forth Translation](https://arxiv.org/abs/2408.04614) — Nguyen et al., 2024
