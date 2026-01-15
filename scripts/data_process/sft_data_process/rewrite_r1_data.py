#!/usr/bin/env python3
"""
Rewrite NQ/HotpotQA dataset into AutoSearch SFT format using Hindsight Generation.
This script reads (question, golden_answer) pairs and generates AutoSearch traces.
"""

import os
import json
import yaml
import time
import random
import re
import argparse
import concurrent.futures
import httpx
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Optional

# Import APIKeyManager from generate_sft_data.py
import sys
# Add parent directory to path to import from generate_sft_data.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_sft_data import APIKeyManager

# ==============================================================================
# Prompt Templates
# ==============================================================================

SYSTEM_PROMPT = """You are an expert annotator for Search Agents.
You will be given a (Question, Golden Answer) pair from a 2018 Wikipedia-based dataset.
Your task is to generate a training trace following the AutoSearch XML protocol.

**Protocol:**
1. `<think>`...reasoning...`</think>`
2. Action: <recall>query</recall> OR <search>query</search>
3. Feedback: <memory>...content...</memory> OR <information>...content...</information>
4. <answer>Golden Answer</answer> (MUST match exactly)

**DECISION LOGIC (Critical):**
- **Use <recall>** if the question asks for:
  * Common knowledge (e.g., "Who wrote Hamlet?", "Capital of France")
  * Well-known facts that a model should know from pre-training
  * Simple definitions or basic concepts
  
- **Use <search>** if the question asks for:
  * Specific entity details (e.g., "Who produced the 2005 film X?")
  * Obscure facts or dates
  * Information that requires looking up specific Wikipedia pages
  * Multi-hop reasoning (e.g., "What is the population of the capital of X?")

**CRITICAL CONSTRAINTS:**
1. The <information> content MUST be realistic Wikipedia-style snippets (as of 2018)
2. The information MUST contain evidence that directly supports the golden answer
3. Do NOT include future events or information beyond 2018
4. The reasoning chain must be logically sound: question -> action -> feedback -> answer
5. The final <answer> MUST exactly match the provided golden answer
6. Keep the answer SHORT - just the entity or exact phrase, no sentences

Output MUST be a single valid JSON object:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

def make_user_prompt(question: str, golden_answer: str) -> str:
    """Generate user prompt for GPT-4"""
    return f"""Input Data:
Question: "{question}"
Golden Answer: "{golden_answer}"

Task:
1. Analyze the question difficulty (common knowledge vs specific detail).
2. Generate the complete trace following AutoSearch protocol.
3. Ensure the final <answer> matches exactly: "{golden_answer}"
4. The simulated <information> or <memory> MUST contain evidence supporting the answer.
5. Remember: This is 2018 Wikipedia knowledge, not current events.
"""

# ==============================================================================
# Validation Functions
# ==============================================================================

def em_match(answer1: str, answer2: str) -> bool:
    """Simple exact match check (case-insensitive, stripped)"""
    return answer1.strip().lower() == answer2.strip().lower()

def validate_rewritten_sample(sample: Dict, original_question: str, golden_answer: str) -> bool:
    """Validate the quality of rewritten sample"""
    try:
        if 'messages' not in sample or len(sample['messages']) < 2:
            return False
        
        content = sample['messages'][1]['content']
        
        # 1. Check answer exists and matches
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if not answer_match:
            return False
        
        extracted_answer = answer_match.group(1).strip()
        
        # Check answer match (allow some flexibility)
        if not em_match(extracted_answer, golden_answer):
            # Allow substring match for multi-word answers
            if golden_answer.lower() not in extracted_answer.lower() and extracted_answer.lower() not in golden_answer.lower():
                return False
        
        # 2. Check reasoning exists
        if '`<think>`' not in content:
            return False
        
        # 3. Check action and feedback consistency
        has_recall = '<recall>' in content and '<memory>' in content
        has_search = '<search>' in content and '<information>' in content
        
        if not (has_recall or has_search):
            return False
        
        # Should not have both recall and search in a single trace (simplified)
        if has_recall and has_search:
            # Check order - recall should come before search if both exist
            recall_pos = content.find('<recall>')
            search_pos = content.find('<search>')
            if recall_pos > search_pos:
                return False
        
        # 4. Check answer length (should be short for EM)
        if len(extracted_answer) > 100:
            # Answer might be too long for EM evaluation
            pass  # Warning but not fail
        
        return True
        
    except Exception as e:
        print(f"[Validation Error] {e}")
        return False

# ==============================================================================
# Rewrite Functions
# ==============================================================================

def rewrite_single_sample(
    key_manager: APIKeyManager,
    question: str,
    golden_answer: str,
    model: str = "gpt-4"
) -> Optional[Dict]:
    """Rewrite a single (question, golden_answer) pair into AutoSearch format"""
    
    user_prompt = make_user_prompt(question, golden_answer)
    
    for attempt in range(3):  # Retry up to 3 times
        try:
            client = key_manager.get_client()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7 + (attempt * 0.1)  # Slightly increase temp on retry
            )
            
            raw_content = response.choices[0].message.content
            
            # Clean up markdown code blocks if present
            if "```json" in raw_content:
                clean_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                clean_content = raw_content.split("```")[1].split("```")[0].strip()
            else:
                clean_content = raw_content
            
            try:
                result = json.loads(clean_content)
            except json.JSONDecodeError as e:
                print(f"[JSON Error] Attempt {attempt+1}: {e}")
                print(f"[JSON Error] Raw content (first 200 chars): {raw_content[:200]}...")
                continue
            
            if validate_rewritten_sample(result, question, golden_answer):
                # Add metadata
                result['meta'] = {
                    'source': 'r1_rewritten',
                    'original_question': question,
                    'golden_answer': golden_answer,
                    'model': model
                }
                return result
            else:
                print(f"[Validation Failed] Attempt {attempt+1} for question: {question[:50]}...")
                
        except Exception as e:
            print(f"[API Error] Attempt {attempt+1}: {e}")
            time.sleep(1)
    
    return None

# ==============================================================================
# Main Execution
# ==============================================================================

def load_questions_from_jsonl(jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load questions and golden answers from JSONL file"""
    samples = []
    
    print(f"Loading questions from {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    question = data.get('question', '').strip()
                    golden_answer = data.get('golden_answer', '').strip()
                    
                    if question and golden_answer:
                        samples.append({
                            'question': question,
                            'golden_answer': golden_answer
                        })
                        
                        if max_samples and len(samples) >= max_samples:
                            break
                except json.JSONDecodeError as e:
                    print(f"[Warning] Failed to parse line: {e}")
                    continue
    
    print(f"Loaded {len(samples)} valid samples")
    return samples

def main():
    parser = argparse.ArgumentParser(description='Rewrite NQ/HotpotQA data into AutoSearch SFT format')
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml with API keys")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with (question, golden_answer) pairs")
    parser.add_argument("--output", type=str, default="data/autosearch_sft_r1_rewritten.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to rewrite (default: all)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent workers")
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="LLM model to use (gpt-4, gpt-4o, etc.)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return
    
    try:
        key_manager = APIKeyManager(args.config)
    except Exception as e:
        print(f"Failed to load API keys: {e}")
        return
    
    # Load questions
    samples = load_questions_from_jsonl(args.input, args.max_samples)
    
    if len(samples) == 0:
        print("Error: No valid samples found in input file")
        return
    
    # Shuffle samples
    random.shuffle(samples)
    
    print(f"Total samples to rewrite: {len(samples)}")
    print(f"Using {args.workers} workers")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    
    # Concurrent rewriting
    results = []
    
    def rewrite_wrapper(sample):
        return rewrite_single_sample(
            key_manager,
            sample['question'],
            sample['golden_answer'],
            args.model
        )
    
    print("\nStarting rewrite process...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(rewrite_wrapper, sample) for sample in samples]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(samples), desc="Rewriting"):
            res = future.result()
            if res:
                results.append(res)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Rewrite Complete!")
    print(f"{'='*60}")
    print(f"Generated: {len(results)}/{len(samples)} valid samples")
    print(f"Success rate: {len(results)/len(samples)*100:.1f}%")
    print(f"Saved to: {args.output}")
    
    # Analyze action distribution
    if results:
        recall_count = 0
        search_count = 0
        for result in results:
            content = result['messages'][1]['content']
            if '<recall>' in content:
                recall_count += 1
            if '<search>' in content:
                search_count += 1
        
        print(f"\nAction Distribution:")
        print(f"  <recall> used: {recall_count} ({recall_count/len(results)*100:.1f}%)")
        print(f"  <search> used: {search_count} ({search_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
