# Instruct-data

指令数据合成方法集合，包含多种从不同来源生成 instruction-response 对的技术。

## 方法目录

| 子目录 | 方法 | 思路 |
|--------|------|------|
| `self-instruct/` | Self-Instruct + Evol-Instruct + Persona | 从种子指令出发，LLM 生成 → 进化 → 多角色实例化 |
| `backtranslation/` | Backtranslation | 从原始语料出发，反向生成对应的指令 |

## self-instruct/

基于 Self-Instruct + Evol-Instruct + Persona 的指令数据合成流水线。

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

快速开始：

```bash
cd self-instruct
pip install -r requirements.txt
python run_pipeline.py --num_instructions 10 --evolve_ratio 0.3
```

## backtranslation/

从高质量语料出发，让 LLM 反向生成对应的指令，再过滤、重写得到训练数据。

```
原始语料 (corpus/)
        │
        ▼  Step 1: 语料收集
        ▼  Step 2: 清洗分段
        ▼  Step 3: 反向翻译（语料 → 指令）
        ▼  Step 4: 质量打分 & 过滤
        ▼  Step 5: 重写响应
        ▼
output/
```

快速开始：

```bash
cd backtranslation
python run_pipeline.py
```
