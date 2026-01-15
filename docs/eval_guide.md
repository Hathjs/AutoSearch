# AutoSearch V2 评估指南

## 概述

`scripts/eval_autosearch_sft.py` 是一个支持多轮交互的评估脚本，可以测试AutoSearch V2模型的搜索和召回能力。

## 功能特性

- ✅ 多轮交互模式（类似 `infer.py`）
- ✅ 支持 `<search>` 动作，可调用检索服务器
- ✅ 支持 `<recall>` 动作（返回内存反馈）
- ✅ 自动检测并处理 `<answer>` 标签
- ✅ 可选禁用搜索功能（用于测试模型行为）

## 使用步骤

### 1. 启动检索服务器（可选）

如果你想要真实的搜索功能，需要先启动检索服务器：

```bash
# 在另一个终端启动检索服务器
conda activate retriever
bash retrieval_launch.sh
```

检索服务器默认监听 `http://127.0.0.1:8000/retrieve`。

### 2. 运行评估脚本

#### 基本用法（启用搜索）

```bash
python scripts/eval_autosearch_sft.py \
    --model_path verl/trainer/checkpoints/autosearch_sft_outdata/global_step_42 \
    --questions "What is the capital of France?" "Who wrote 1984?"
```

#### 禁用搜索（仅测试模型行为）

```bash
python scripts/eval_autosearch_sft.py \
    --model_path verl/trainer/checkpoints/autosearch_sft_outdata/global_step_42 \
    --questions "What is the capital of France?" \
    --disable_search
```

#### 指定检索服务器URL

```bash
python scripts/eval_autosearch_sft.py \
    --model_path verl/trainer/checkpoints/autosearch_sft_outdata/global_step_42 \
    --questions "What is the capital of France?" \
    --search_url http://127.0.0.1:8000/retrieve
```

#### 使用测试文件

```bash
python scripts/eval_autosearch_sft.py \
    --model_path verl/trainer/checkpoints/autosearch_sft_outdata/global_step_42 \
    --test_file data/test_questions.jsonl
```

### 3. 参数说明

- `--model_path`: 训练好的模型路径（必需）
- `--questions`: 测试问题列表（空格分隔）
- `--test_file`: 包含测试问题的JSONL文件（每行一个JSON，包含 `question` 字段）
- `--max_new_tokens`: 每轮生成的最大token数（默认：1024）
- `--temperature`: 采样温度（默认：0.7）
- `--search_url`: 检索服务器URL（默认：`http://127.0.0.1:8000/retrieve`）
- `--max_turns`: 最大交互轮数（默认：10）
- `--disable_search`: 禁用搜索API调用（用于测试模型行为）

## 工作流程

1. **加载模型**：加载训练好的SFT模型
2. **多轮交互**：
   - 模型生成响应直到遇到 `</search>`, `</recall>`, 或 `</answer>`
   - 如果检测到 `<search>`，调用检索服务器并返回 `<information>...</information>`
   - 如果检测到 `<recall>`，返回 `<memory>Retrieving internal knowledge...</memory>`
   - 如果检测到 `<answer>`，结束交互
3. **解析结果**：解析所有XML标签并统计
4. **保存结果**：结果保存到 `eval_results.json`

## 输出示例

```
================================================================================
Test 1/2: What is the capital of France?
================================================================================

[Starting multi-turn interaction (max 10 turns)]

--- Turn 1 ---
Generated: <think>The user is asking about the capital of France...
[Model requested recall: 'capital of France']
--- Turn 2 ---
Generated: <memory>Paris is the capital of France.</memory>
<answer>Paris</answer>
[Final answer detected, ending interaction]

================================================================================
Generated Response:
<think>...</think>
<recall>capital of France</recall>
<memory>Paris is the capital of France.</memory>
<answer>Paris</answer>
================================================================================
Parsed Results:
  - Num Turns: 2
  - Has Reasoning: True
  - Has Recall: True (queries: ['capital of France'])
  - Has Memory: True
  - Has Search: False (queries: [])
  - Has Information: False
  - Has Answer: True
  - Answer: Paris
```

## 常见问题

### Q: 评估脚本卡住了怎么办？Ctrl+C 无法中断？

A: 
**如果程序在生成过程中卡住（没有输出）：**

1. **第一次尝试**：按 `Ctrl+C` 一次，等待 5-10 秒
   - 程序现在有信号处理，应该能响应 Ctrl+C
   - 如果看到 `[Generation interrupted by user]` 说明成功中断

2. **如果 Ctrl+C 无效**：
   ```bash
   # 在另一个终端查找进程
   ps aux | grep eval_autosearch_sft
   
   # 强制终止（替换 PID 为实际进程ID）
   kill -9 <PID>
   ```

3. **预防措施**：
   - 使用较小的 `--max_new_tokens`（如 256 或 128）
   - 使用 `--disable_search` 先测试模型行为
   - 检查GPU内存是否足够（`nvidia-smi`）

4. **调试建议**：
   - 程序现在会输出详细的调试信息，包括：
     - `[Tokenizing prompt...]` 
     - `[Generating... This may take 10-60 seconds...]`
     - `[Generation completed...]`
   - 如果卡在某个步骤，可以通过这些信息定位问题

### Q: 生成速度很慢怎么办？

A:
1. 减少 `--max_new_tokens`：默认是 512，可以试试 256 或 128
2. 使用较低的 `--temperature`（如 0.1）减少采样时间
3. 检查GPU利用率：`nvidia-smi`
4. 确保没有其他程序占用GPU

### Q: 如何测试没有检索服务器的情况？

A: 使用 `--disable_search` 参数，模型会收到占位符响应。

### Q: 检索服务器启动失败怎么办？

A: 
1. 检查是否正确安装了依赖（`faiss-gpu`, `transformers` 等）
2. 检查索引文件和语料文件路径是否正确
3. 参考 `docs/retriever.md` 查看详细配置

## 参考

- 检索服务器配置：`docs/retriever.md`
- 检索服务器启动脚本：`retrieval_launch.sh`
- Search-R1推理示例：`infer.py`
