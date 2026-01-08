# AutoSearch SFT 调试输出示例

## 预期调试输出格式

当运行 `scripts/data_process/autosearch_sft.py --debug` 时，会输出类似以下的调试信息：

```
================================================================================
DEBUG: Tokenization and Label Masking Visualization
================================================================================

[System Prompt + User Query] (MASKED - Loss=0):
--------------------------------------------------------------------------------
You are AutoSearch, an intelligent agent with dual information retrieval capabilities.
Please answer the user's question step-by-step.

**Core Protocol & Action Space:**
1. **Reasoning (`<think>`):** - ALWAYS start with `<think>` to analyze the question.
   - Assess if you have sufficient internal knowledge.

2. **Internal Recall (`<recall>`):** - **PRIORITY:** If the question asks for static facts, definitions, or common sense, verify your memory first.
   - *Syntax:* `<recall>keywords</recall>`
   - *System Feedback:* Returns `<memory>...result...</memory>`
   - Use this for facts, definitions, and knowledge you should know.

3. **External Search (`<search>`):** - **SECONDARY:** Use ONLY if internal memory is insufficient, uncertain (`No record`), or if the question requires real-time/private info.
   - *Syntax:* `<search>keywords</search>`
   - *System Feedback:* Returns `<information>...results...</information>`
   - Use this for real-time data, news, or information not in your training data.

4. **Termination:**
   - Output final answer in `<answer>...content...</answer>`.

**Trap/Hallucination Handling:**
- If `<recall>` returns `<memory>No record</memory>` or contradicts the user's premise, you MUST refuse to answer or correct the premise. Do NOT fabricate information.

Question: What is the capital of France?

[Assistant Response] (with Label Masking):
--------------------------------------------------------------------------------
[Token    1] <think>          | Label:  151645 | [TRAIN]
[Token    2] This                           | Label:  151645 | [TRAIN]
[Token    3]  is                           | Label:  151645 | [TRAIN]
[Token    4]  a                            | Label:  151645 |[TRAIN]
[Token    5]  basic                        | Label:  151645 | [TRAIN]
...
[Token   15] </think>         | Label:  151645 | [TRAIN]
[Token   16] <recall>                      | Label:  151645 | [TRAIN]
[Token   17] capital                       | Label:  151645 | [TRAIN]
[Token   18]  of                           | Label:  151645 | [TRAIN]
[Token   19]  France                       | Label:  151645 | [TRAIN]
[Token   20] </recall>                     | Label:  151645 | [TRAIN]
[Token   21] <memory>                      | Label:     -100 | [MASKED] | <memory> content
[Token   22] Paris                         | Label:     -100 | [MASKED] | <memory> content
[Token   23]  is                           | Label:     -100 | [MASKED] | <memory> content
[Token   24]  the                          | Label:     -100 | [MASKED] | <memory> content
[Token   25]  capital                      | Label:     -100 | [MASKED] | <memory> content
[Token   26]  of                           | Label:     -100 | [MASKED] | <memory> content
[Token   27]  France                       | Label:     -100 | [MASKED] | <memory> content
[Token   28] .                             | Label:     -100 | [MASKED] | <memory> content
[Token   29] </memory>                     | Label:     -100 | [MASKED] | <memory> content
[Token   30] <answer>                      | Label:  151645 | [TRAIN]
[Token   31] Paris                         | Label:  151645 | [TRAIN]
[Token   32]  is                           | Label:  151645 | [TRAIN]
[Token   33]  the                          | Label:  151645 | [TRAIN]
[Token   34]  capital                      | Label:  151645 | [TRAIN]
[Token   35]  of                           | Label:  151645 | [TRAIN]
[Token   36]  France                       | Label:  151645 | [TRAIN]
[Token   37] .                             | Label:  151645 | [TRAIN]
[Token   38] </answer>                     | Label:  151645 | [TRAIN]

--------------------------------------------------------------------------------
FINAL SUMMARY:
  Total tokens: 250
  Tokens to train (loss computed): 180
  Tokens masked (loss=0): 70
  Masking ratio: 28.00%
================================================================================
```

## 关键验证点

### 1. System Prompt 和 User Query 完全被 Mask
- 所有在 `assistant_start_char` 之前的 token 的 label 都应该是 `-100`
- 这些 token 不会出现在 "[Assistant Response]" 部分

### 2. `<memory>...</memory>` 内容被 Mask
- `<memory>` 标签内的所有内容（包括标签本身）的 label 都应该是 `-100`
- 这确保了模型不会学习预测系统反馈

### 3. 其他部分正常训练
- `<think>...</think>`: Label = token_id (训练)
- `<recall>...</recall>`: Label = token_id (训练)
- `<search>...</search>`: Label = token_id (训练)
- `<information>...</information>`: Label = token_id (训练)
- `<answer>...</answer>`: Label = token_id (训练)

## 三种测试场景的预期

### Case 1: Recall Success
- `<think>`: [TRAIN]
- `<recall>capital of France</recall>`: [TRAIN]
- `<memory>Paris is the capital of France.</memory>`: [MASKED] ⚠️
- `<answer>Paris is the capital of France.</answer>`: [TRAIN]

### Case 2: Recall Fail/Trap
- `<think>`: [TRAIN]
- `<recall>2024 Turing Award winner</recall>`: [TRAIN]
- `<memory>No record found.</memory>`: [MASKED] ⚠️
- `<think>` (第二次): [TRAIN]
- `<answer>The 2024 Turing Award winner has not been announced yet.</answer>`: [TRAIN]

### Case 3: External Search
- `<think>`: [TRAIN]
- `<search>AAPL stock price today</search>`: [TRAIN]
- `<information>AAPL is currently trading at $220.50...</information>`: [TRAIN]
- `<answer>As of today, Apple (AAPL) stock is trading at $220.50...</answer>`: [TRAIN]

注意：`<information>` 内容**不应该**被 mask，因为它是外部搜索的结果，模型需要学习如何使用这些信息。


