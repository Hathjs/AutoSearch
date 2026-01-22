# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess AutoSearch SFT dataset from JSONL to Parquet format with tokenization and label masking.

Input: JSONL file with messages format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "<think>...</think><recall>...</recall><memory>...</memory><answer>...</answer>"}
  ]
}

Output: Parquet file with tokenized input_ids, labels, and attention_mask.
"""

import re
import os
import json
import datasets
from transformers import AutoTokenizer
import numpy as np
import argparse

# Optional import for HDFS support
try:
    from verl.utils.hdfs_io import copy, makedirs
    HDFS_AVAILABLE = True
except ImportError:
    HDFS_AVAILABLE = False
    # Define dummy functions if verl is not available
    def copy(*args, **kwargs):
        raise ImportError("verl module not available. Cannot copy to HDFS. Install verl or skip --hdfs_dir option.")
    def makedirs(*args, **kwargs):
        raise ImportError("verl module not available. Cannot create HDFS directories. Install verl or skip --hdfs_dir option.")


def make_prefix(question, template_type='base'):
    """
    Generate system prompt for AutoSearch V2 with dual-channel architecture.
    
    Args:
        question: User question string
        template_type: Template type (currently only 'base' supported)
    
    Returns:
        Formatted prompt string
    """
    if template_type == 'base':
        """AutoSearch V2 protocol with recall and search channels"""
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


def find_system_feedback_spans(text):
    """
    Find all system feedback spans in the text.
    Includes both <memory>...</memory> and <information>...</information>.
    
    Returns:
        List of (start_char, end_char) tuples for each block
    """
    spans = []
    
    # Pattern for memory blocks
    for match in re.finditer(r'<memory>(.*?)</memory>', text, re.DOTALL):
        spans.append((match.start(), match.end()))
        
    # Pattern for information blocks (search results)
    for match in re.finditer(r'<information>(.*?)</information>', text, re.DOTALL):
        spans.append((match.start(), match.end()))
        
    return spans


def create_label_mask_with_offsets(input_ids, offset_mapping, full_text, assistant_start_char):
    """
    Create label mask for SFT training using pre-computed offset mapping.
    
    Masking rules:
    - System prompt + User query: -100 (no loss)
    - <memory>...</memory> content: -100 (no loss, system feedback)
    - Everything else (reasoning, recall, search, answer): keep token id (compute loss)
    
    Args:
        input_ids: Tokenized input IDs (numpy array)
        offset_mapping: Tokenizer offset mapping (list of tuples)
        full_text: Complete text including system prompt and assistant response
        assistant_start_char: Character position where assistant response starts
    
    Returns:
        labels: Numpy array of label IDs (same length as input_ids)
    """
    # Initialize labels with input_ids (will mask later)
    labels = input_ids.copy()
    
    # Find system feedback spans (memory + information)
    feedback_spans = find_system_feedback_spans(full_text)
    
    # Mask 1: System prompt + User query (everything before assistant response)
    # offset_mapping is a list of (char_start, char_end) tuples
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        # Handle special tokens (offset is None or (0, 0))
        if char_start is None or char_end is None:
            # Special tokens at the beginning should be masked
            if token_idx == 0:
                labels[token_idx] = -100
            continue
        
        # Mask tokens that are entirely before assistant response
        if char_end <= assistant_start_char:
            labels[token_idx] = -100
    
    # Mask 2: System feedback content blocks
    for start_char, end_char in feedback_spans:
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None or char_end is None:
                continue
            
            # Check if this token overlaps with feedback span
            # Token overlaps if it's not completely before or after the span
            if not (char_end <= start_char or char_start >= end_char):
                labels[token_idx] = -100
    
    return labels


def debug_print_tokenization(tokenizer, input_ids, labels, full_text, assistant_start_char, offset_mapping):
    """
    Debug function to visualize tokenization and label masking.
    
    Args:
        tokenizer: Tokenizer instance
        input_ids: Token IDs
        labels: Label IDs (with -100 for masked tokens)
        full_text: Original full text
        assistant_start_char: Character position where assistant response starts
        offset_mapping: Token offset mapping
    """
    print("\n" + "="*80)
    print("DEBUG: Tokenization and Label Masking Visualization")
    print("="*80)
    
    # Decode tokens
    decoded_tokens = []
    for token_id in input_ids:
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        decoded_tokens.append(token_str)
    
    # Find feedback spans
    feedback_spans = find_system_feedback_spans(full_text)
    
    # Print system prompt section
    print("\n[System Prompt + User Query] (MASKED - Loss=0):")
    print("-" * 80)
    system_text = full_text[:assistant_start_char]
    print(system_text[:200] + "..." if len(system_text) > 200 else system_text)
    
    # Print assistant response with masking visualization
    print("\n[Assistant Response] (with Label Masking):")
    print("-" * 80)
    
    current_pos = 0
    for i, (token_id, label, offset) in enumerate(zip(input_ids, labels, offset_mapping)):
        char_start, char_end = offset
        
        # Skip special tokens at the beginning
        if char_start is None or char_end is None:
            if i == 0:
                print(f"[Token {i:4d}] {decoded_tokens[i]:30s} | Label: {label:6d} | [SPECIAL TOKEN - MASKED]")
            continue
        
        # Check if this is in system prompt
        if char_end <= assistant_start_char:
            continue  # Already printed above
        
        # Check if this token is masked
        is_masked = (label == -100)
        mask_status = "[MASKED]" if is_masked else "[TRAIN]"
        
        # Get the actual text for this token
        token_text = full_text[char_start:char_end] if char_start < len(full_text) else decoded_tokens[i]
        
        # Check if this is part of feedback
        in_feedback = False
        tag_type = ""
        for start, end in feedback_spans:
            if not (char_end <= start or char_start >= end):
                in_feedback = True
                # Identify tag type for display
                span_text = full_text[start:end]
                if span_text.startswith("<memory>"):
                    tag_type = "memory"
                elif span_text.startswith("<information>"):
                    tag_type = "information"
                break
        
        # Format output
        if in_feedback:
            print(f"[Token {i:4d}] {token_text[:30]:30s} | Label: {label:6d} | {mask_status} | <{tag_type}> content")
        else:
            print(f"[Token {i:4d}] {token_text[:30]:30s} | Label: {label:6d} | {mask_status}")
        
        # Print summary every 50 tokens
        if (i + 1) % 50 == 0:
            masked_count = sum(1 for l in labels[:i+1] if l == -100)
            train_count = i + 1 - masked_count
            print(f"  ... Summary: {train_count} tokens to train, {masked_count} tokens masked")
    
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


def process_sft_example(example, tokenizer, template_type='base', max_length=4096, debug=False, use_chat_template=True):
    """
    Process a single SFT example from JSONL format with chat_template support.
    
    Args:
        example: Dict with 'messages' key containing conversation
        tokenizer: AutoTokenizer instance
        template_type: Template type for prompt generation
        max_length: Maximum sequence length
        debug: Whether to print debug information
        use_chat_template: Whether to use chat_template (default: True)
    
    Returns:
        Dict with 'input_ids', 'labels', 'attention_mask', 'messages' (for chat_template)
    """
    messages = example['messages']
    
    # Extract user query and assistant response
    user_content = None
    assistant_content = None
    
    for msg in messages:
        if msg['role'] == 'user':
            user_content = msg['content']
        elif msg['role'] == 'assistant':
            assistant_content = msg['content']
    
    if user_content is None or assistant_content is None:
        raise ValueError("Missing user or assistant message in example")
    
    # Build messages in chat format for chat_template
    # Format: [{"role": "user", "content": system_prompt + question}, {"role": "assistant", "content": response}]
    system_prompt = make_prefix(user_content, template_type=template_type)
    
    if use_chat_template and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Use chat_template format (for RL training compatibility)
        chat_messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # Apply chat_template to get formatted text
        formatted_text = tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=False,  # Don't add generation prompt, we have full conversation
            tokenize=False
        )
        
        # Tokenize with offset mapping
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors='pt',
            add_special_tokens=False,  # chat_template already includes special tokens
            return_offsets_mapping=True
        )
        
        # Find where assistant response starts in the formatted text
        # The formatted text should have user content, then assistant content
        # We need to find the boundary after user content ends
        user_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": system_prompt}],
            add_generation_prompt=True,  # This adds the assistant start token
            tokenize=False
        )
        assistant_start_char = len(user_formatted)
        
        full_text = formatted_text
        
    else:
        # Fallback: No chat_template (original behavior)
        full_text = system_prompt + assistant_content
        assistant_start_char = len(system_prompt)
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors='pt',
            add_special_tokens=True,
            return_offsets_mapping=True
        )
    
    input_ids = tokenized['input_ids'][0].numpy()
    attention_mask = tokenized['attention_mask'][0].numpy()
    offset_mapping = tokenized['offset_mapping'][0]
    
    # Create labels with masking using the same tokenization
    labels = create_label_mask_with_offsets(
        input_ids,
        offset_mapping,
        full_text,
        assistant_start_char
    )
    
    # Debug output
    if debug:
        debug_print_tokenization(
            tokenizer, input_ids, labels, full_text, 
            assistant_start_char, offset_mapping
        )
    
    # Return tokenized data and chat format for SFTDataset
    result = {
        'input_ids': input_ids.tolist(),
        'labels': labels.tolist(),
        'attention_mask': attention_mask.tolist(),
        # Also save original text format for backward compatibility
        'question': user_content,
        'answer': assistant_content,
    }
    
    # Save chat format for SFTDataset (if using chat_template)
    if use_chat_template and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Save messages as JSON string (Parquet doesn't support nested dicts well)
        import json
        result['messages'] = json.dumps(chat_messages)
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process AutoSearch SFT dataset from JSONL to Parquet')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSONL file path')
    parser.add_argument('--output_dir', type=str, default='./data/autosearch_sft',
                        help='Output directory for parquet files')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output parquet file name (default: auto-detect from input filename)')
    parser.add_argument('--model_name', type=str, 
                        default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/model/Qwen2.5-3B-Instruct',
                        help='Tokenizer model name or path')
    parser.add_argument('--template_type', type=str, default='base',
                        help='Prompt template type')
    parser.add_argument('--hdfs_dir', type=str, default=None,
                        help='Optional HDFS directory to copy results to')
    parser.add_argument('--max_length', type=int, default=4096,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for first example')
    parser.add_argument('--use_chat_template', action='store_true', default=True,
                        help='Use chat_template for formatting (default: True)')
    parser.add_argument('--no_chat_template', dest='use_chat_template', action='store_false',
                        help='Disable chat_template (use original format)')
    
    args = parser.parse_args()
    
    # Load tokenizer (local files only, disable HuggingFace Hub)
    model_path = os.path.abspath(os.path.expanduser(args.model_name))
    print(f"Loading tokenizer from local path: {model_path}...")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path is not a directory: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True  # Disable HuggingFace Hub, use local files only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load JSONL file
    print(f"Loading JSONL from {args.input_file}...")
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} examples")
    
    # Process examples
    print("Processing examples...")
    processed_data = []
    for idx, example in enumerate(data):
        try:
            # Enable debug for first example if --debug flag is set
            debug_mode = args.debug and idx == 0
            if debug_mode:
                print(f"\n{'='*80}")
                print(f"DEBUG MODE: Processing example {idx + 1}")
                print(f"{'='*80}")
                print(f"User: {example['messages'][0]['content'] if example['messages'][0]['role'] == 'user' else 'N/A'}")
                print(f"Assistant: {example['messages'][1]['content'][:100]}..." if len(example['messages']) > 1 else 'N/A')
            
            processed = process_sft_example(
                example, 
                tokenizer, 
                args.template_type,
                max_length=args.max_length,
                debug=debug_mode,
                use_chat_template=args.use_chat_template
            )
            processed_data.append(processed)
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(data)} examples")
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Successfully processed {len(processed_data)} examples")
    
    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_list(processed_data)
    
    # Determine output filename
    if args.output_file:
        output_filename = args.output_file
    else:
        # Auto-detect from input filename
        input_basename = os.path.basename(args.input_file).lower()
        if 'val' in input_basename or 'test' in input_basename:
            output_filename = 'val.parquet'
        elif 'train' in input_basename:
            output_filename = 'train.parquet'
        else:
            # Default to train if cannot determine
            output_filename = 'train.parquet'
            print(f"[Warning] Cannot determine split type from input filename, defaulting to train.parquet")
    
    # Save to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, output_filename)
    print(f"Saving to {output_file}...")
    dataset.to_parquet(output_file)
    
    print(f"Saved {len(processed_data)} examples to {output_file}")
    
    # Optional: Copy to HDFS
    if args.hdfs_dir is not None:
        if not HDFS_AVAILABLE:
            print(f"[Warning] verl module not available. Skipping HDFS copy.")
            print(f"[Warning] To enable HDFS support, install verl module or skip --hdfs_dir option.")
        else:
            try:
                makedirs(args.hdfs_dir)
                copy(src=args.output_dir, dst=args.hdfs_dir)
                print(f"Copied to HDFS: {args.hdfs_dir}")
            except Exception as e:
                print(f"[Error] Failed to copy to HDFS: {e}")

