# Recall Tool 使用指南

## 概述

Recall Tool 是 AutoSearch V2 框架中的核心组件，用于实现内部知识回忆（Internal Knowledge Recall）。它通过专门的 prompt 引导模型从参数化记忆中"想起"知识，从而将知识回忆与逻辑推理解耦。

## 设计理念

### 1. 解耦推理和知识回忆

- **之前（Search-R1）**：`<think>` 混合了逻辑推理和知识回忆
- **现在（AutoSearch V2）**：
  - `<think>` 只负责逻辑推理
  - `<recall>` 专门负责知识回忆
  - 减少 Sycophancy（阿谀奉承）问题

### 2. 参数化记忆

- 模型通过训练已经学习了常见知识（存储在模型权重中）
- `<recall>` 工具通过专门的 prompt 激发模型"想起"这些知识
- 不是学习新知识，而是引导模型回忆已有知识

### 3. 作为独立工具

- `<recall>` 和 `<search>` 都是工具调用
- `<recall>`：内部回忆，成本为 0
- `<search>`：外部搜索，成本为 -0.1（RL 惩罚）

## 实现架构

### 核心模块

**`search_r1/search/recall_tool.py`**

提供两个主要函数：

1. **`recall_internal_knowledge()`**：单个查询的回忆
2. **`batch_recall_internal_knowledge()`**：批量查询的回忆

### 工作流程

```
1. 模型生成: <recall>capital of France</recall>
2. 检测到 recall 动作
3. 调用 recall_internal_knowledge(query="capital of France", ...)
4. 使用专门的 prompt 引导模型回忆
5. 模型生成: <memory>The capital of France is Paris.</memory>
6. 返回给主流程，继续推理
```

## 使用场景

### 1. 评估脚本 (`scripts/eval_autosearch_sft.py`)

```python
from search_r1.search.recall_tool import recall_internal_knowledge

# 检测到 <recall> 动作时
memory_content = recall_internal_knowledge(
    query=recall_query_text,
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=256,
    temperature=0.1
)
```

### 2. RL 训练 (`search_r1/llm_agent/generation.py`)

在 `execute_predictions()` 中：

- **验证模式**：调用 recall 工具，测试模型回忆能力
- **训练模式**：使用占位符，模型基于 SFT 训练自己生成记忆内容

### 3. 推理脚本 (`infer.py`)

支持 `<recall>` 和 `<search>` 两种动作，实现完整的双通道架构。

## 配置参数

### `recall_internal_knowledge()`

- `query`: 回忆查询（从 `<recall>query</recall>` 中提取）
- `model`: 模型实例
- `tokenizer`: tokenizer 实例
- `device`: 设备（"cuda" 或 "cpu"）
- `max_new_tokens`: 最大生成 token 数（默认 256）
- `temperature`: 采样温度（默认 0.1，低温度更确定）
- `use_chat_template`: 是否使用 chat template（默认 True）

## 与 Search-R1 的区别

| 特性 | Search-R1 | AutoSearch V2 |
|------|-----------|---------------|
| 推理方式 | `<think>` 混合推理和知识 | `<think>` 只推理，`<recall>` 专门回忆 |
| 知识获取 | 只有 `<search>` 外部搜索 | `<recall>` 内部回忆 + `<search>` 外部搜索 |
| 解耦程度 | 未解耦 | 完全解耦 |
| Sycophancy 问题 | 容易在推理中编造 | 通过分离减少问题 |
| RL 训练 | 难以区分"记错"和"推错" | 可以单独奖励/惩罚 recall |

## 最佳实践

### 1. 评估时

- 启用 recall 工具（`do_recall=True`）
- 使用低温度（0.1）确保确定性回忆
- 检查模型是否能正确回忆常见知识

### 2. RL 训练时

- 验证模式：启用 recall 工具测试
- 训练模式：使用占位符，让模型基于 SFT 训练生成
- 在奖励函数中区分 recall 和 search 的成本

### 3. Prompt 设计

Recall prompt 应该：
- 明确指示模型回忆训练数据中的知识
- 如果不知道，返回 "No record found"
- 保持简洁，避免干扰模型回忆

## 故障排除

### 问题：Recall 工具无法访问模型

**原因**：在 RL 训练中，模型访问路径可能不同

**解决**：
- 检查 `actor_rollout_wg` 的结构
- 在验证模式下使用 recall 工具
- 在训练模式下使用占位符

### 问题：模型回忆的内容不准确

**原因**：模型可能没有在训练数据中学习到相关知识

**解决**：
- 检查 SFT 数据是否包含相关知识
- 调整 recall prompt
- 增加相关训练数据

### 问题：Recall 和 Search 混淆

**原因**：模型没有学会区分何时使用 recall vs search

**解决**：
- 检查 SFT 数据中的动作分布
- 确保 Type A（recall）和 Type B（search）数据平衡
- 在 prompt 中明确区分使用场景

## 未来改进

1. **更智能的 Recall**：可以基于查询类型选择不同的 recall prompt
2. **Recall 缓存**：缓存常见查询的回忆结果
3. **Recall 质量评估**：评估回忆内容的准确性
4. **混合 Recall+Search**：支持先 recall 后 search 的混合策略
