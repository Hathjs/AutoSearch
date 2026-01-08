#!/usr/bin/env python3
"""
Standalone debug script for AutoSearch SFT tokenization and label masking.
This script doesn't require full dependencies, just transformers.
"""

import re
import json
from transformers import AutoTokenizer
import numpy as np


def make_prefix(question, template_type='base'):
    """Generate system prompt for AutoSearch V2."""
    if template_type == 'base':
        prefix = f"""You are AutoSearch, an intelligent agent with dual information retrieval capabilities.
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

Question: {question}\n"""
    else:
        raise NotImplementedError(f"Template type {template_type} not supported")
    return prefix


def find_memory_spans(text):
    """Find all <memory>...</memory> spans in the text."""
    pattern = r'<memory>(.*?)</memory>'
    spans = []
    for match in re.finditer(pattern, text, re.DOTALL):
        spans.append((match.start(), match.end()))
    return spans


def create_label_mask_with_offsets(input_ids, offset_mapping, full_text, assistant_start_char):
    """Create label mask for SFT training."""
    labels = input_ids.copy()
    memory_spans = find_memory_spans(full_text)
    
    # Mask 1: System prompt + User query
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start is None or char_end is None:
            if token_idx == 0:
                labels[token_idx] = -100
            continue
        if char_end <= assistant_start_char:
            labels[token_idx] = -100
    
    # Mask 2: Memory content blocks
    for mem_start_char, mem_end_char in memory_spans:
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None or char_end is None:
                continue
            if not (char_end <= mem_start_char or char_start >= mem_end_char):
                labels[token_idx] = -100
    
    return labels


def debug_print_tokenization(tokenizer, input_ids, labels, full_text, assistant_start_char, offset_mapping):
    """Debug function to visualize tokenization and label masking."""
    print("\n" + "="*80)
    print("DEBUG: Tokenization and Label Masking Visualization")
    print("="*80)
    
    # Decode tokens
    decoded_tokens = []
    for token_id in input_ids:
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        decoded_tokens.append(token_str)
    
    # Find memory spans
    memory_spans = find_memory_spans(full_text)
    
    # Print system prompt section
    print("\n[System Prompt + User Query] (MASKED - Loss=0):")
    print("-" * 80)
    system_text = full_text[:assistant_start_char]
    print(system_text[:300] + "..." if len(system_text) > 300 else system_text)
    
    # Print assistant response with masking visualization
    print("\n[Assistant Response] (with Label Masking):")
    print("-" * 80)
    
    # Group tokens by sections for better visualization
    sections = []
    current_section = {"name": "", "tokens": [], "masked": False}
    
    for i, (token_id, label, offset) in enumerate(zip(input_ids, labels, offset_mapping)):
        char_start, char_end = offset
        
        if char_start is None or char_end is None:
            if i == 0:
                sections.append({"name": "[SPECIAL TOKEN]", "tokens": [i], "masked": True})
            continue
        
        if char_end <= assistant_start_char:
            continue  # System prompt, already printed
        
        # Check if masked
        is_masked = (label == -100)
        
        # Check if in memory
        in_memory = False
        for mem_start, mem_end in memory_spans:
            if not (char_end <= mem_start or char_start >= mem_end):
                in_memory = True
                break
        
        # Get token text
        token_text = full_text[char_start:char_end] if char_start < len(full_text) else decoded_tokens[i]
        
        # Detect section boundaries
        if '<think>' in token_text or '</think>' in token_text:
            if current_section["tokens"]:
                sections.append(current_section)
            current_section = {"name": "[THINK]", "tokens": [i], "masked": is_masked}
        elif '<recall>' in token_text or '</recall>' in token_text:
            if current_section["tokens"]:
                sections.append(current_section)
            current_section = {"name": "[RECALL]", "tokens": [i], "masked": is_masked}
        elif in_memory:
            if current_section["name"] != "[MEMORY]":
                if current_section["tokens"]:
                    sections.append(current_section)
                current_section = {"name": "[MEMORY]", "tokens": [i], "masked": True}
            else:
                current_section["tokens"].append(i)
        elif '<search>' in token_text or '</search>' in token_text:
            if current_section["tokens"]:
                sections.append(current_section)
            current_section = {"name": "[SEARCH]", "tokens": [i], "masked": is_masked}
        elif '<information>' in token_text or '</information>' in token_text:
            if current_section["tokens"]:
                sections.append(current_section)
            current_section = {"name": "[INFORMATION]", "tokens": [i], "masked": is_masked}
        elif '<answer>' in token_text or '</answer>' in token_text:
            if current_section["tokens"]:
                sections.append(current_section)
            current_section = {"name": "[ANSWER]", "tokens": [i], "masked": is_masked}
        else:
            current_section["tokens"].append(i)
            if is_masked:
                current_section["masked"] = True
    
    if current_section["tokens"]:
        sections.append(current_section)
    
    # Print sections
    for section in sections:
        section_name = section["name"]
        masked_status = "[MASKED]" if section["masked"] else "[TRAIN]"
        print(f"\n{section_name} {masked_status}")
        print("-" * 80)
        
        # Print first few and last few tokens of this section
        token_indices = section["tokens"]
        if len(token_indices) <= 10:
            for idx in token_indices:
                char_start, char_end = offset_mapping[idx]
                token_text = full_text[char_start:char_end] if char_start < len(full_text) else decoded_tokens[idx]
                label = labels[idx]
                mask_status = "[MASKED]" if label == -100 else "[TRAIN]"
                print(f"  Token {idx:4d}: {repr(token_text[:40]):40s} | Label: {label:6d} | {mask_status}")
        else:
            for idx in token_indices[:5]:
                char_start, char_end = offset_mapping[idx]
                token_text = full_text[char_start:char_end] if char_start < len(full_text) else decoded_tokens[idx]
                label = labels[idx]
                mask_status = "[MASKED]" if label == -100 else "[TRAIN]"
                print(f"  Token {idx:4d}: {repr(token_text[:40]):40s} | Label: {label:6d} | {mask_status}")
            print(f"  ... ({len(token_indices) - 10} more tokens) ...")
            for idx in token_indices[-5:]:
                char_start, char_end = offset_mapping[idx]
                token_text = full_text[char_start:char_end] if char_start < len(full_text) else decoded_tokens[idx]
                label = labels[idx]
                mask_status = "[MASKED]" if label == -100 else "[TRAIN]"
                print(f"  Token {idx:4d}: {repr(token_text[:40]):40s} | Label: {label:6d} | {mask_status}")
    
    # Final summary
    masked_count = sum(1 for l in labels if l == -100)
    train_count = len(labels) - masked_count
    print("\n" + "-" * 80)
    print(f"FINAL SUMMARY:")
    print(f"  Total tokens: {len(labels)}")
    print(f"  Tokens to train (loss computed): {train_count}")
    print(f"  Tokens masked (loss=0): {masked_count}")
    print(f"  Masking ratio: {masked_count/len(labels)*100:.2f}%")
    print("="*80 + "\n")


def main():
    # Load sample data
    with open('data/autosearch_sft_sample.jsonl', 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process first example
    example = examples[0]
    print(f"\n{'='*80}")
    print(f"Processing Example 1: {example['messages'][0]['content']}")
    print(f"{'='*80}")
    
    user_content = example['messages'][0]['content']
    assistant_content = example['messages'][1]['content']
    
    system_prompt = make_prefix(user_content)
    full_text = system_prompt + assistant_content
    assistant_start_char = len(system_prompt)
    
    # Tokenize
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors='pt',
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    
    input_ids = tokenized['input_ids'][0].numpy()
    offset_mapping = tokenized['offset_mapping'][0]
    
    # Create labels
    labels = create_label_mask_with_offsets(
        input_ids,
        offset_mapping,
        full_text,
        assistant_start_char
    )
    
    # Debug print
    debug_print_tokenization(
        tokenizer, input_ids, labels, full_text,
        assistant_start_char, offset_mapping
    )


if __name__ == '__main__':
    main()


