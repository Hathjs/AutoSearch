# SFT Data Processing Scripts

This directory contains scripts for processing and generating SFT training data for AutoSearch V2.

## Overview

The data generation pipeline consists of several steps:

1. **Extract R1 Questions** (`extract_r1_questions.py`): Extract questions and golden answers from Search-R1 parquet files
2. **Rewrite R1 Data** (`rewrite_r1_data.py`): Convert NQ/HotpotQA data into AutoSearch format using Hindsight Generation
3. **Filter Old Data** (`filter_old_data.py`): Filter existing SFT data, keeping only Type A and Type C
4. **Merge SFT Data** (`merge_sft_data.py`): Combine rewritten data with filtered old data

## Scripts

### 1. extract_r1_questions.py

Extract questions and golden answers from Search-R1 parquet files.

**Usage:**
```bash
python scripts/data_process/sft_data_process/extract_r1_questions.py \
    --input /path/to/train.parquet \
    --output data/r1_questions.jsonl \
    --max_samples 5000
```

**Arguments:**
- `--input`: Input parquet file path (required)
- `--output`: Output JSONL file path (required)
- `--max_samples`: Maximum number of samples to extract (optional)

### 2. rewrite_r1_data.py

Rewrite NQ/HotpotQA questions into AutoSearch SFT format using GPT-4.

**Usage:**
```bash
python scripts/data_process/sft_data_process/rewrite_r1_data.py \
    --config config.yaml \
    --input data/r1_questions.jsonl \
    --output data/autosearch_sft_r1_rewritten.jsonl \
    --max_samples 5000 \
    --workers 10 \
    --model gpt-4
```

**Arguments:**
- `--config`: Path to config.yaml with API keys (default: config.yaml)
- `--input`: Input JSONL file with (question, golden_answer) pairs (required)
- `--output`: Output JSONL file path (default: data/autosearch_sft_r1_rewritten.jsonl)
- `--max_samples`: Maximum number of samples to rewrite (optional)
- `--workers`: Number of concurrent workers (default: 10)
- `--model`: LLM model to use (default: gpt-4)
- `--seed`: Random seed (default: 42)

### 3. filter_old_data.py

Filter old SFT data, keeping only Type A (Internal Recall) and Type C (Trap/Hallucination).

**Usage:**
```bash
python scripts/data_process/sft_data_process/filter_old_data.py \
    --input data/autosearch_sft_raw.jsonl \
    --output data/autosearch_sft_filtered.jsonl \
    --keep_types A C
```

**Arguments:**
- `--input`: Input JSONL file with old SFT data (required)
- `--output`: Output JSONL file path (required)
- `--keep_types`: Types to keep (default: A C)

### 4. merge_sft_data.py

Merge multiple SFT data files into a single dataset.

**Usage:**
```bash
python scripts/data_process/sft_data_process/merge_sft_data.py \
    --inputs data/autosearch_sft_r1_rewritten.jsonl \
             data/autosearch_sft_filtered.jsonl \
    --output data/autosearch_sft_merged.jsonl \
    --shuffle \
    --seed 42
```

**Arguments:**
- `--inputs`: Input JSONL files (can specify weight like "file.jsonl:2" for 2x sampling)
- `--output`: Output JSONL file path (required)
- `--shuffle`: Shuffle the merged data (default: True)
- `--no_shuffle`: Do not shuffle
- `--seed`: Random seed for shuffling (default: 42)

**Example with weights:**
```bash
python scripts/data_process/sft_data_process/merge_sft_data.py \
    --inputs data/r1_rewritten.jsonl:1 \
             data/type_a.jsonl:2 \
             data/type_c.jsonl:1 \
    --output data/merged.jsonl
```

## Complete Pipeline Example

```bash
# Step 1: Extract questions from Search-R1 data
python scripts/data_process/sft_data_process/extract_r1_questions.py \
    --input /home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/chenqing30/cqproject/searchr1-rag-traindata/nq_search/train.parquet \
    --output data/r1_questions.jsonl \
    --max_samples 5000

# Step 2: Rewrite questions into AutoSearch format
python scripts/data_process/sft_data_process/rewrite_r1_data.py \
    --config config.yaml \
    --input data/r1_questions.jsonl \
    --output data/autosearch_sft_r1_rewritten.jsonl \
    --max_samples 5000 \
    --workers 10

# Step 3: Filter old data (keep Type A and C)
python scripts/data_process/sft_data_process/filter_old_data.py \
    --input data/autosearch_sft_raw.jsonl \
    --output data/autosearch_sft_filtered.jsonl \
    --keep_types A C

# Step 4: Merge all data
python scripts/data_process/sft_data_process/merge_sft_data.py \
    --inputs data/autosearch_sft_r1_rewritten.jsonl \
             data/autosearch_sft_filtered.jsonl \
    --output data/autosearch_sft_final.jsonl \
    --shuffle \
    --seed 42
```

## Data Formats

### Input Format (for rewrite_r1_data.py)

```json
{
  "question": "What is the capital of France?",
  "golden_answer": "Paris",
  "index": 0,
  "data_source": "nq"
}
```

### Output Format

```json
{
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "<think>...</think><recall>...</recall><memory>...</memory><answer>Paris</answer>"
    }
  ],
  "meta": {
    "source": "r1_rewritten",
    "original_question": "What is the capital of France?",
    "golden_answer": "Paris",
    "model": "gpt-4"
  }
}
```

## Notes

- All scripts output statistics about the processed data
- The rewrite script includes validation to ensure quality
- The merge script supports weighted sampling for data balancing
- All scripts preserve metadata for tracking data sources
