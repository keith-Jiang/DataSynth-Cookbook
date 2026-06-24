# Data Synthesis

## 数据合成
### Instruct 数据

#### Self-Instruct
> https://arxiv.org/abs/2212.10560
> https://github.com/yizhongw/self-instruct

Self-Instruct 多样化指令，分为以下四个步骤：
1. 从人工编写的 175 条高质量的 seed instructions 出发，每条指令代表一种不同的任务类型（如摘要、翻译、问答等）
2.  从种子指令池中随机采样少量指令作为 few-shot 示例，利用 LLM 的 ICL 能力生成新指令，每一轮生成的新指令会被加入指令池中，供后续轮次继续采样使用，形成一个不断扩展的自举（bootstrapping）循环
3. 模型为每条新指令生成对应的回答，从而构成完整的训练样本
4. 对生成的数据进行过滤，剔除低质量、重复或与已有指令过于相似的样本

<img src="assets/data_synthesis/self_instruct.png" alt="self_instruct" style="zoom: 100%;" />

#### Evol-Instruct
> https://arxiv.org/abs/2304.12244

Self-Instruct 产生的指令多样性和复杂度有限，Evol-Instruct 在此基础上从目前已有简单指令出发做进化与增强，分为以下两个进化方向：

1. **深度进化**：让指令变得更难、更复杂，包括：
   * 增加约束：给任务添加更多限制条件
   * 深化：要求更深层次的思考或推理
   * 具体化：将泛泛的问题变得更具体
   * 增加推理步骤：让解决问题需要更多步骤
   * 复杂化输入：使输入内容本身更复杂

   例如，从"写一个排序函数"进化到"写一个排序函数，要求时间复杂度 `O(nlogn)`，支持自定义比较器，能处理含有 None 值的列表，并且是稳定排序"
2. **广度进化**：生成一个全新的、主题不同但难度相当的指令，目的是增加多样性
例如，从"写一个排序函数"进化到"设计一个简单的缓存系统"

<img src="assets/data_synthesis/evol_instruct.png" alt="evol_instruct" style="zoom: 100%;" />

### Persona Hub
> https://arxiv.org/abs/2406.20094
> https://github.com/tencent-ailab/persona-hub

Persona Hub 构建了一个包含10亿个多样化 Persona 的数据库，当生成数据时先随机采样一个 Persona，然后让 LLM 扮演该 Persona 来生成合成数据，保证了数据的多样性

<img src="assets/data_synthesis/persona.png" alt="persona" style="zoom: 100%;" />

Persona 的构造分为两种：
* **Text-Persona**：从大规模网页语料出发，利用 LLM 从每段文本中推断出可能写出/阅读/关心这段文本的人的 Persona 描述
* **Persona-Persona**：从一个Persona 衍生出相关但不同的 Persona

### CoT 数据

#### Self-Consistency
> https://arxiv.org/pdf/2203.11171

Self-Consistency 对 CoT 进行多次采样，最终取答案一致的 CoT 样本作为 SFT 的训练样本

<img src="assets/data_synthesis/self_consistency.png" alt="persona" style="zoom: 100%;" />

#### STaR
> https://arxiv.org/abs/2203.14465

STaR 让模型尝试用 CoT 推理解题，对于模型做错的题，把正确答案作为提示让模型重新生成推理过程，最终保留答案正确的 CoT 微调模型，然后再利用微调后更强的模型生成回答，实现自循环

*可能出现忠实性幻觉——答案正确但是 CoT 错误*

<img src="assets/data_synthesis/star.png" alt="star" style="zoom: 100%;" />

### 垂域数据

#### PMC-LLaMA
> https://arxiv.org/abs/2304.14454

PMC-LLaMA 在 4.8M 生物医学学术论文和 30K 医学教材上进行领域预训练，并构建了 202M tokens 的指令微调数据，覆盖医学问答、推理及对话任务

*   医学 QA 数据合成：设置 Seed Instruction 进行多样性改写
*   医学 CoT 数据合成：将现有的医学多选题输入大模型，并生成从问题到答案的完整 CoT，以及逐个分析选项对错的题目 CoT
*   使用 UMLS（统一医学语言系统）知识图谱：构造形如“请解释 \[实体\] 的定义”的问答对，构造形如“确定 \[实体A\] 与 \[实体B\] 之间的关系”的问答对

<img src="assets/data_synthesis/pmc_llama.png" alt="pmc_llama" style="zoom: 100%;" />

RAG 数据库构建清晰，可以 doc 加描述，Pre-retrieve 进行 query 改写与子 query 分解，Retrieve 进行混合检索（向量+稀疏+图），然后 Rerank，定位文档保证可追溯

#### LawGPT
> https://arxiv.org/abs/2502.06572
> https://github.com/LAMDA-NeSy/Knowledge-Guide-Data-Generation

LawGPT 提出知识引导数据生成框架（KgDG），利用法律知识体系（法条、案例、法律概念）引导 LLM 生成多样化的法律 QA 与推理数据，并通过精炼与验证流程保证质量

1.  **Knowledge-Guided Generation**：

    *   Knowledge-Aware Sampler：

        1.  根据 Seed Problem，让模型判断需要哪类法律知识（如刑事或民事）

        2.  从知识库中采样真实的法律文书（DOCS），确保生成的案例具有真实的专业背景

    *   Knowledge-Guided Writer：

        1.  模型根据采样的法律文书原型，编造新的案情、姓名、地点（去隐私化）

        2.  强制要求生成 `reasoning`（推理过程）和 `reference`（引用的具体法条字典）字段，而不仅仅是答案

2.  **Knowledge-Guide Fixer**：

    *   Reference Modifier：将生成的法条 Key 对应的内容（Value）与知识库中的标准法条内容进行比对和替换，确保引用内容 100% 准确

    *   Reasoning Corrector：检查推理路径中的逻辑错误或计算错误（例如涉案金额的累加是否正确），并输出校正后的 JSON

3.  **Data Verifier**：

    *   一致性校验：验证修正后的推理路径、法条依据与最终答案之间是否逻辑自洽

    *   过滤机制：如果模型判断推理过程与答案依然不匹配或存在无法修复的错误，则该条数据被标记为“错误”并丢弃

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/d117f863-b216-41c1-9431-1470fc3d7431.png)

#### TRIDENT

TRIDENT \[[https://arxiv.org/abs/2505.24672](https://arxiv.org/abs/2505.24672)\] 提出三维度多样化红队数据合成框架，沿**词汇多样性**、**恶意意图**和**越狱策略**三个轴进行扩展，pipeline 具体如下

1.  **定义意图领域 (Intent Domains)**：基于 LlamaGuard-3 和 MLCommons 标准，确定了 14 个恶意意图类别（如暴力犯罪、诽谤、代码解释器滥用等）

2.  **情境生成 (Scenario Generation)**：使用无审查的 Llama-3.1-8B 模型，针对每个意图领域生成具体的情境

3.  **Persona 生成与扩展**：

    *   **情境转角色 (Scenario-to-Persona)**：从情境中推断出角色的职业、性格特征、观点和生活经历（例如：一个通过技术手段操纵他人的黑客）

    *   **角色扩展 (Persona-to-Persona)**：基于“六度分隔理论”扩展相关角色（例如：从黑客扩展到其网络安全顾问或受影响的开发者）

4.  **指令生成 (Instruction Generation)**：

    *   **角色扮演生成**：让模型扮演上述角色，生成带有其独特语言风格和特定恶意意图的指令

    *   **引入越狱策略 (Jailbreak Tactics)**：通过 **TRIDENT-EDGE** 流程，应用 6 种越狱方法（密文编码、代码注入、低资源语言翻译、过去式改写、角色调制、RENELLM 变体）对指令进行改写

5.  **回复生成 (Response Generation)**：使用 **GPT-4o-mini** 配合专门的 **CoT 模板**生成既符合安全标准（不直接参与有害行为）又具有帮助性（解释危害并提供合规建议）的回复

*可以用于内容安全数据合成*

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/de7dbf54-18aa-4ce0-b610-e73142008a27.png)

## Code 数据合成

代码数据合成的核心挑战在于生成功能正确、逻辑多样且可通过执行验证的代码数据。区别于自然语言，代码数据天然具备**可执行验证性**，因此"合成-执行-过滤"闭环成为该领域的主流范式

### WizardCoder

WizardCoder \[[https://arxiv.org/abs/2306.08568](https://arxiv.org/abs/2306.08568)\] 将 Evol-Instruct 方法适配到代码领域，提出 Code Evol-Instruct，数据来源于 Code Alpaca，并应用 GPT3.5 进行进化，针对代码任务设计了特定的进化算子，从简单编码指令逐步进化出高复杂度的代码任务

1.  **添加约束和要求**：引入额外的规范，如长度限制或字符类型要求

2.  **用具体要求替换通用要求**：鼓励使用特定算法或数据结构

3.  **添加推理步骤**：对更简单的问题强制要求更复杂的逻辑过程

4.  **引入错误代码**：提供不正确的代码片段以测试鲁棒性和错误识别能力

5.  **提出效率要求**：要求更高的时间或空间复杂度优化

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/490c133b-1506-4d2b-8112-28abda11454e.png)

### Magicoder / OSS-Instruct

Magicoder \[[https://arxiv.org/abs/2312.02120](https://arxiv.org/abs/2312.02120)\] 提出 OSS-Instruct，使用**开源代码片段**作为种子引导 LLM 生成指令数据，分为

1.  **种子采集：** 从开源语料库（如 `starcoderdata`）中随机抽取 1-15 行连续的代码片段。

2.  **激发灵感：** 将这些片段输入给教师模型（Teacher Model），并使用特定的 Prompt 要求模型：“以此代码片段为灵感，创造一个高质量的编程问题及其完整解决方案。”

3.  **多样化生成：** 由于开源代码涵盖了各种库、算法和实际应用场景（如 Shell 脚本、机器学习库调用、类定义等），生成的数据比传统的 `Self-Instruct` 更具多样性和现实性。

4.  **数据清洗与去重：** 排除重复样本，并进行严格的**数据脱毒（Decontamination）**，剔除与 HumanEval、MBPP 等评测集重叠的代码，最终形成约 75K 条指令数据

随后进行 SFT

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/51dc6f97-4ff3-43e5-a9a0-9e8490bb28a7.png)

### KodCode

KodCode \[[https://arxiv.org/abs/2503.02951](https://arxiv.org/abs/2503.02951)\] 构建了三阶段流水线：

1.  **编码问题合成 (Coding Question Synthesis)**：

    *   **基础编码：** 使用 **MAGPIE-Prefill** 方法，通过预填提示词（如 "Write a Python function that..."）引导模型（Qwen2.5-Coder-7B）自动补全出多样化的基础任务

    *   **编码考核：** 以 LeetCode、Codeforces、APPS 和 TACO 等高质量人类数据集为种子，利用 GPT-4o 作为“老师”生成类似难度和结构的新问题

    *   **算法与数据结构 (DSA)：** 从 GitHub 开源库中采样 DSA 代码片段，转换成考察基础理解的评估问题

    *   **技术文档转化：** 将 Flask、Pandas、PyTorch 等主流库的文档转化为具体的问题，测试模型对特定包的掌握程度

    *   **大规模扩展：** 利用 MAGPIE 框架配合多个开源模型生成大量原始数据，并经过严格的分类过滤

2.  **解法与测试生成 (Solution & Test Generation)**：

    *   **自验证机制：** 使用 GPT-4o 同时生成解决方案和单元测试，并在 Python 环境中实际运行。

    *   **分支覆盖率：** 利用 `pytest-cov` 框架分析测试用例，只有通过验证且达到 **100% 分支覆盖率** 的三元组（问题-方案-测试）才会被保留。

    *   **难题重试策略：** 针对模型难以一次性写对的“难题”，不直接丢弃，而是分配最多 **10次尝试机会**。每次尝试都会重新生成方案和测试。这种方法自然地通过“尝试次数”为题目贴上了难度标签（Easy/Medium/Hard）

3.  **后训练数据合成 (Post-training Data Synthesis)**：

    *   **格式多样化：** 使用样式转换器将 NL（自然语言）问题改写为函数补全、工具调用等多种格式。

    *   **推理增强 (CoT)：** 使用 **DeepSeek-R1** 为所有问题生成思维链（Chain-of-Thought）回复。为了保证思维链的正确性，采用基于测试的**拒绝采样 (Reject Sampling)**：对每个问题生成3个 R1 结果，只保留能通过单元测试的那个

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/3430c37e-d909-43ef-88fb-8d5798d09957.png)

## Agent trace

Agent 数据合成的核心在于生成高质量的**交互轨迹**（Trajectory），包括推理步骤、工具调用、环境观察和奖励反馈。与纯文本数据不同，Agent 数据天然具有多轮、多模态、环境交互的特性

### AgentInstruct (Microsoft)

AgentInstruct \[[https://arxiv.org/abs/2407.03502](https://arxiv.org/abs/2407.03502)\] 自动从原始文档（如文本、代码）中创建高质量、多样化的合成数据。其核心流程包含三个主要步骤：

*   **内容转换流 (Content Transformation Flow)**：将原始种子数据（如教科书章节、网页文章、代码片段）转换为中间表示形式。例如，将代码片段转换为 API 描述，或从非结构化文本中提取关键知识点，以便后续生成指令

*   **指令生成流 (Seed Instruction Creation Flow)**：由一组指令生成智能体根据转换后的内容，创建多样化的初始指令。该框架包含一个拥有超过 100 个子类别的分类法，确保覆盖不同的技能领域（如推理、数学、创意写作等）

*   **精炼流 (Refinement Flow)**：利用另一组精炼智能体对生成的指令进行迭代改进。通过自我反思（Reflection）和工具辅助（如搜索、代码解释器），提升指令的复杂度和响应的质量

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/906ea62f-d763-46cd-aa47-77063a2df798.png)

### FireAct

FireAct \[[https://arxiv.org/abs/2310.05915](https://arxiv.org/abs/2310.05915)\] 提出从强模型蒸馏 Agent 轨迹的范式：使用 GPT-4 在多个任务和多种 prompting 方法上生成任务求解轨迹，统一转换为 ReAct 格式用于微调小模型。仅用 500 条 GPT-4 轨迹微调 Llama2-7B，在 HotpotQA 上性能提升 77%（14.8% → 26.2%），Llama2-13B 提升 62%。多任务多方法的多样化训练数据显著优于单一来源

### AgentTuning

AgentTuning \[[https://arxiv.org/abs/2310.12823](https://arxiv.org/abs/2310.12823)\] 构建了 AgentInstruct 数据集，包含 6 种 Agent 任务（操作系统、数据库、知识图谱、网页导航、家务操持、网购）的高质量交互轨迹。采用混合指令微调策略，将 Agent 轨迹与通用指令数据结合训练，在保持通用 LLM 能力的同时赋予 Agent 能力。AgentLM-70B 在未见任务上表现与 GPT-3.5 相当（+76% 提升）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/a3031bdf-0476-4860-9bf9-b419005dde11.png)

### AgentTrek

AgentTrek \[[https://arxiv.org/abs/2412.09605](https://arxiv.org/abs/2412.09605)\] 提出从网页教程自动合成 GUI Agent 轨迹的流水线：

1.  **自动教程采集与处理**：

    *   **来源与过滤**：从大规模语料库（如 RedPajama）中提取与 GUI 操作相关的文本。利用 **FastText 分类器** 过滤掉无关内容，仅保留具有“教程特征”的文本。

    *   **结构化转换**：使用 LLM（如 GPT-4o-mini）将非结构化文本转换为标准格式，包含：**高层目标**、**前置条件**和**分步操作指南**

2.  **引导回放生成轨迹 (Guided Replay)**：

    *   **执行环境**：在真实的 Web 浏览器环境（BrowserGym/Chromium）中运行

    *   **VLM 引导执行**：由一个强大的视觉语言模型（VLM）充当“执行者”，根据结构化教程的步骤，一步步在真实网页上操作

    *   **多模态记录**：实时记录操作过程中的所有数据，包括：

        *   **视觉**：屏幕截图、视频录像

        *   **文本/语义**：HTML 代码、可访问性树 (AXTree)

        *   **认知**：智能体在每一步生成的 **CoT 推理**

        *   **动作**：具体的 Playwright API 调用或像素级坐标

    *   **质量验证**：

        *   **自动评估**：引入专门的 VLM 评估器，根据任务描述、执行动作和最终截图来判断该轨迹是否成功完成目标。只有通过验证的高质量轨迹才会加入训练集

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/648924ae-47da-49b7-829b-a4e4f99e3c79.png)

### ProductResearch

阿里国际提出的数据合成框架 \[[https://arxiv.org/abs/2602.23716](https://arxiv.org/abs/2602.23716)\]，为了缓解电商领域中 Deep Research 的 Domain Gap，具体如

* 现有模型在处理复杂购物咨询时深度不足

* 目前的模型用于 Deep Research 时在电商工具调用上泛化性差

* 缺乏高质量的长程（Long-horizon）的电商研究轨迹数据

  于是，文章提出了如下数据合成框架：

* 多智能体合成：使用 User Agent、Research Agent、 Supervisor Agent 协同生成高质量轨迹

  *   User Agent (用户智能体) —— 模拟需求

      *   输入： 真实的匿名用户历史行为序列（购买、评论、咨询记录）

      *   输出：

          *   画像提取： 从历史中推断用户的购物偏好和专业程度

          *   查询构造： 生成一个复杂的、无法通过简单搜索解决的深度研究问题（例如：“为极地摄影选择一套兼顾防护与画质的专业相机系统”）

          *   动态 RACE Rubric 生成： 为每个查询生成个性化的评分标准，规定“全面性、深度、指令遵循、可读性”四个维度的权重

  *   Research Agent (研究智能体) —— 执行任务

      *   采用 Plan → Toolcall → Report 的操作模式，利用双环境工具，包括开放网页搜索（Serper API/Crawl4AI）和内部电商库查询（BM25 检索/商品详情查找），在监督下不断迭代，直到生成一份证据确凿、对比详尽的研究报告

  *   Supervisor Agent (监督智能体) —— 质量把关

      *   Check Plan (计划检查)： 检查研究策略是否逻辑严密、覆盖全面

      *   Check Toolcall (工具调用检查)： 验证参数是否正确、信息是否有用、是否陷入死循环

      *   Check Report (报告检查)： 严格对照 User Agent 生成的 Rubric，检查是否有证据支持、是否包含 3-5 个真实商品、是否有幻觉

* 反思内化 (Reflective Internalization)：要求回顾整个批评-反馈-修改的流程，将多轮的“监督-修改”对话蒸馏为单角色带有 CoT 的训练样本

* 微调: 在合成的高保真轨迹上对 MoE 模型（如 Qwen3-30B-A3B）进行指令微调

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb5RbVvrqXx/img/39f65ca5-086f-4214-81d7-86baf948cafa.png)

但是，该方法只适用于生产单轮 Deep Research 数据

### SPAG

SPAG（Self-Playing Adversarial Language Game）\[NeurIPS 2024\] 通过对抗性语言博弈（Adversarial Taboo）进行自博弈 RL 训练。博弈中模型同时扮演攻击者和防御者角色，通过博弈结果作为 RL 信号训练。模型的推理能力在各基准上均匀提升，且迭代自博弈实现了推理能力的持续增强

### Language Self-Play（LSP）

Language Self-Play \[[https://arxiv.org/abs/2509.07414](https://arxiv.org/abs/2509.07414)\]（Meta/UC Berkeley 2025）将博弈论中的自博弈范式引入 LLM 训练。一个模型扮演 **Challenger**（生成对 Solver 困难但有解的指令），另一个扮演 **Solver**（努力解决挑战）。两者相互对抗推动彼此进化，无需额外训练数据，在指令遵循、数学推理和编码任务上均有提升

### DTE（Debate, Train, Evolve）

DTE \[[https://arxiv.org/abs/2505.15734](https://arxiv.org/abs/2505.15734)\]（EMNLP 2025）利用多 Agent **辩论轨迹**在无真实标签下进化单一模型。提出 Reflect-Critique-Refine prompting 策略提升辩论质量，通过辩论过程中的推理轨迹提取训练信号。在 GSM-PLUS 上平均提升 8.92%，并展现 5.8% 的跨领域泛化增益

### SPELL

SPELL \[[https://arxiv.org/abs/2509.23863](https://arxiv.org/abs/2509.23863)\]（2025）提出多角色自博弈 RL 框架，单一模型同时扮演**提问者**、**回答者**和**验证者**三个角色进行自我交互。引入自动化课程学习控制文档长度难度，以及自适应奖励函数根据训练进展动态调整。在推理基准上平均提升 7.6 点（含 Qwen3-30B）

### T-SPIN

T-SPIN \[[https://arxiv.org/abs/2601.08198](https://arxiv.org/abs/2601.08198)\]（2025）针对 SPIN 训练不稳定的问题，引入**三元组结构**：在当前策略生成、初始策略生成和真实数据之间建立历史优势关系，并加入熵约束实现无参考模型的稳定微调。仅需 25% 的样本量即可达到标准 SFT 同等性能

### SPACE

SPACE \[[https://arxiv.org/abs/2512.07175](https://arxiv.org/abs/2512.07175)\]（2025）使用**噪声对比估计**（NCE）稳定 Self-Play 优化过程。将合成样本视为辅助分量，通过二分类方式区分合成/真实样本，确保优化的稳定收敛，避免了 SPIN 系列方法中常见的训练震荡问题