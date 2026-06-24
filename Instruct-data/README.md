# Instruct-data

基于 Self-Instruct + Evol-Instruct + Persona 的指令数据合成流水线。

## 流水线

```
seed_tasks.jsonl (175条种子指令)
        │
        ▼  Step 1: 指令生成
        │   few-shot 采样种子指令 → LLM 生成新指令 → ROUGE-L 去重
        ▼
machine_generated_instructions.jsonl
        │
        ▼  Step 2: 指令进化 (Evol-Instruct)
        │   In-depth:  加约束 / 加深推理 / 具体化 / 拆子任务
        │   In-breadth: 换主题生成变体
        ▼
evolved_instructions.jsonl
        │
        ▼  Step 3: 实例生成 (Persona)
        │   28种角色随机分配 → 不同风格的 input-output
        ▼
machine_generated_instances.jsonl
        │
        ▼  Step 4: 数据整理
        │   解析 / 清洗 / 去重 / 合并种子数据
        ▼
final_training_data.jsonl
```

## 快速开始
claude --resume 6c9b749e-b131-4b05-91b1-a38dd1a7c82e    
```bash
pip install -r requirements.txt

# 小规模测试（10条指令、30%进化）
python run_pipeline.py --num_instructions 10 --evolve_ratio 0.3

# 完整跑（100条指令、包含进化指令和种子数据）
python run_pipeline.py \
    --num_instructions 100 \
    --evolve_ratio 0.5 \
    --include_evolved \
    --include_seed_tasks
```

## 单独跑某一步

```bash
python run_pipeline.py --skip_to 2    # 从 Step 2 开始（跳过 Step 1）
python run_pipeline.py --skip_to 3    # 从 Step 3 开始
```

每个 step 也可以独立运行：

```bash
python step1_generate_instructions.py --num_instructions_to_generate 50
python step2_evolve_instructions.py --evolve_ratio 0.5
python step3_generate_instances.py --include_evolved
python step4_prepare_data.py --include_seed_tasks
```

## 配置

在 `configs.py` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_INSTRUCTIONS_TO_GENERATE` | 100 | Step 1 生成指令数 |
| `ROUGE_THRESHOLD` | 0.7 | 去重相似度阈值 |
| `EVOLVE_RATIO` | 0.5 | 多少比例的指令进入进化 |
| `EVOLVE_IN_DEPTH_RATIO` | 0.6 | 进化中 in-depth 占比 |
| `EVOLVE_BREADTH_COUNT` | 2 | 每条指令 in-breadth 变体数 |
| `PERSONA_TEMPERATURE` | 0.3 | 实例生成温度 |
| `REQUEST_BATCH_SIZE` | 5 | 每批 API 请求数 |

## 输出格式

`final_training_data.jsonl` 每行：

```json
{
    "instruction": "Write a Python function to...",
    "input": "list of integers [1, 2, 3]",
    "output": "def process(lst):\n    return ...",
    "persona": "a senior software engineer"
}
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `step1_generate_instructions.py` | 从种子指令 few-shot 生成新指令 |
| `step2_evolve_instructions.py` | Evol-Instruct：in-depth 加复杂度 + in-breadth 扩展多样性 |
| `step3_generate_instances.py` | 随机 persona 驱动生成 input-output 对 |
| `step4_prepare_data.py` | 解析清洗去重，输出最终训练数据 |
| `run_pipeline.py` | 串联 4 步的入口脚本 |
| `personas.py` | 28 个角色定义（学术/技术/创意/教育/日常） |
| `api_utils.py` | DeepSeek API 封装（支持重试退避） |
| `configs.py` | 全局参数配置 |
| `seed_tasks.jsonl` | 175 条人工编写的种子指令 |
