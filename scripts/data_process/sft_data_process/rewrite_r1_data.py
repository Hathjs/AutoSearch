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
import threading
import httpx
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Optional, Tuple

# Import APIKeyManager from generate_sft_data.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_sft_data import APIKeyManager

# ==============================================================================
# Prompt Templates
# ==============================================================================

SYSTEM_PROMPT_BASE = """You are an expert annotator for Search Agents.
You will be given a (Question, Golden Answer) pair from a 2018 Wikipedia-based dataset.
Your task is to generate a training trace following the AutoSearch XML protocol.

**Protocol:**
1. `<think>`...reasoning...`</think>`
2. Action: <recall>query</recall> OR <search>query</search>
3. Feedback: <memory>...content...</memory> OR <information>...content...</information>
4. <answer>Golden Answer</answer> (MUST match exactly)

**DECISION LOGIC (Critical):**
- **Use <recall>** if the question asks for:
  * Well-known facts that most people know (e.g., "Capital of France" → Paris)
  * Common knowledge from pre-training (e.g., "Who wrote Hamlet?" → Shakespeare)
  * Basic definitions (e.g., "What is photosynthesis?" → process by which plants...)
  * Simple historical facts (e.g., "When did World War II end?" → 1945)
  * **Rule**: If you would know this without looking it up, use <recall>
  
- **Use <search>** if the question asks for:
  * Specific dates, numbers, or statistics (e.g., "Population of X in 1990")
  * Obscure facts or recent events (e.g., "Who produced the 2005 film X?")
  * Information requiring Wikipedia lookup (e.g., "Who played character Y in movie Z?")
  * Specific entity details not in common knowledge
  * **Rule**: If you need to look up specific information, use <search>

**CRITICAL CONSTRAINTS:**
1. **TAG PAIRING RULES (MUST FOLLOW):**
   - If you use <recall>, you MUST use <memory> for feedback (NOT <information>)
   - If you use <search>, you MUST use <information> for feedback (NOT <memory>)
   - DO NOT mix: <recall> with <information> or <search> with <memory>

2. **ANSWER TAG (MANDATORY):**
   - You MUST include an <answer> tag with the exact golden answer
   - Format: <answer>Golden Answer</answer>
   - The answer MUST be SHORT - just the entity or exact phrase, no sentences
   - Example: <answer>Paris</answer> NOT <answer>The capital is Paris.</answer>

3. **Content Quality:**
   - The <information> content MUST be realistic Wikipedia-style snippets (as of 2018)
   - The <memory> content MUST be concise facts that the model should know
   - The information MUST contain evidence that directly supports the golden answer
   - Do NOT include future events or information beyond 2018

4. **Reasoning Chain:**
   - The reasoning chain must be logically sound: question -> action -> feedback -> answer
   - The final <answer> MUST exactly match the provided golden answer

Output MUST be a single valid JSON object:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

**REMINDER: Your output MUST include <answer>Golden Answer</answer> tag!**

**FEW-SHOT (FORMAT ONLY; DO NOT COPY FACTS VERBATIM):**
- Single-hop + recall:
{"messages":[{"role":"user","content":"EXAMPLE_Q1"},{"role":"assistant","content":"<think>Common knowledge → recall.</think>\n<recall>EXAMPLE_QUERY</recall>\n<memory>EXAMPLE_EVIDENCE_CONTAINING_ANSWER</memory>\n<answer>EXAMPLE_ANSWER</answer>"}]}
- Single-hop + search:
{"messages":[{"role":"user","content":"EXAMPLE_Q2"},{"role":"assistant","content":"<think>Specific detail → search.</think>\n<search>EXAMPLE_QUERY</search>\n<information>EXAMPLE_WIKI_SNIPPET_CONTAINING_ANSWER</information>\n<answer>EXAMPLE_ANSWER</answer>"}]}
"""

# Additional prompt for multi-hop questions (HotpotQA)
SYSTEM_PROMPT_MULTIHOP = """You are an expert annotator for Search Agents.
You will be given a (Question, Golden Answer) pair from a 2018 Wikipedia-based dataset.
This is a MULTI-HOP question that requires reasoning across multiple facts or entities.

Your task is to generate a training trace following the AutoSearch XML protocol.

**Protocol:**
1. `<think>`...reasoning...`</think>`
2. Action: <recall>query</recall> OR <search>query</search>
3. Feedback: <memory>...content...</memory> OR <information>...content...</information>
4. [If needed] Additional reasoning and search/recall steps for multi-hop reasoning
5. <answer>Golden Answer</answer> (MUST match exactly)

**MULTI-HOP REASONING LOGIC:**
- Multi-hop questions require connecting multiple pieces of information
- You may need to:
  1. First search/recall for intermediate information (e.g., "What is the capital of X?")
  2. Then use that information to search/recall for the final answer (e.g., "What is the population of [capital]?")
- Each step should have its own `<think>`, action, and feedback
- The final answer should be derived from the combined information

**DECISION LOGIC:**
- **Use <recall>** for common knowledge that the model should know
- **Use <search>** for specific facts, dates, or obscure information
- For multi-hop questions, you may need MULTIPLE search/recall steps

**CRITICAL CONSTRAINTS:**
1. **TAG PAIRING RULES (MUST FOLLOW):**
   - If you use <recall>, you MUST use <memory> for feedback (NOT <information>)
   - If you use <search>, you MUST use <information> for feedback (NOT <memory>)
   - DO NOT mix: <recall> with <information> or <search> with <memory>
   - Each step must follow this rule consistently

2. **ANSWER TAG (MANDATORY):**
   - You MUST include an <answer> tag with the exact golden answer
   - Format: <answer>Golden Answer</answer>
   - The answer MUST be SHORT - just the entity or exact phrase, no sentences

3. **Content Quality:**
   - The <information> content MUST be realistic Wikipedia-style snippets (as of 2018)
   - The <memory> content MUST be concise facts that the model should know
   - Each information snippet should support one step of the reasoning chain
   - Do NOT include future events or information beyond 2018

4. **Reasoning Chain:**
   - The reasoning chain must be logically sound and show the multi-hop progression
   - The final <answer> MUST exactly match the provided golden answer

Output MUST be a single valid JSON object:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

**REMINDER: Your output MUST include <answer>Golden Answer</answer> tag!**

**FEW-SHOT (MULTI-HOP SHAPE; DO NOT COPY FACTS VERBATIM):**
{"messages":[{"role":"user","content":"EXAMPLE_MULTI_HOP_Q"},{"role":"assistant","content":"<think>Need intermediate fact then final fact.</think>\n<search>EXAMPLE_QUERY_STEP1</search>\n<information>EXAMPLE_SNIPPET_STEP1_CONTAINING_INTERMEDIATE</information>\n<think>Use intermediate to find final answer.</think>\n<search>EXAMPLE_QUERY_STEP2</search>\n<information>EXAMPLE_SNIPPET_STEP2_CONTAINING_FINAL_ANSWER</information>\n<answer>EXAMPLE_ANSWER</answer>"}]}
"""

def get_system_prompt(data_source: str) -> str:
    """Get appropriate system prompt based on data source"""
    if data_source == 'hotpotqa':
        return SYSTEM_PROMPT_MULTIHOP
    else:
        return SYSTEM_PROMPT_BASE

def make_user_prompt(
    question: str,
    golden_answer: str,
    data_source: str = 'nq',
    desired_action: Optional[str] = None,
) -> str:
    """Generate user prompt for GPT-4"""
    base_prompt = f"""Input Data:
Question: "{question}"
Golden Answer: "{golden_answer}"

**CRITICAL REQUIREMENTS:**
1. You MUST include <answer>{golden_answer}</answer> in your output
2. Follow tag pairing rules: <recall> -> <memory>, <search> -> <information>
3. Do NOT mix: <recall> with <information> or <search> with <memory>
"""

    if desired_action in ("recall", "search"):
        base_prompt += f"""
**ACTION OVERRIDE (MANDATORY):**
- You MUST choose <{desired_action}> as the FIRST action.
- Still follow tag pairing rules and include <answer>{golden_answer}</answer>.
"""
    
    if data_source == 'hotpotqa':
        base_prompt += f"""
Task:
1. This is a MULTI-HOP question - analyze what intermediate information is needed.
2. Generate the complete trace with multiple reasoning steps if needed.
3. Each step should have: <think> -> action -> feedback
4. Ensure the final <answer> matches exactly: "{golden_answer}"
5. The simulated <information> or <memory> MUST contain evidence supporting each step.
6. Remember: This is 2018 Wikipedia knowledge, not current events.
7. **MANDATORY**: Include <answer>{golden_answer}</answer> at the end.
"""
    else:
        base_prompt += f"""
Task:
1. Analyze the question difficulty (common knowledge vs specific detail).
   - If it's common knowledge (e.g., "Capital of France"), use <recall> -> <memory>
   - If it requires specific lookup (e.g., "Who produced film X in 2005?"), use <search> -> <information>
2. Generate the complete trace following AutoSearch protocol.
3. Ensure the final <answer> matches exactly: "{golden_answer}"
4. The simulated <information> or <memory> MUST contain evidence supporting the answer.
5. Remember: This is 2018 Wikipedia knowledge, not current events.
6. **MANDATORY**: Include <answer>{golden_answer}</answer> at the end.
"""
    
    return base_prompt

# ==============================================================================
# Validation Functions
# ==============================================================================

def em_match(answer1: str, answer2: str) -> bool:
    """Simple exact match check (case-insensitive, stripped)"""
    return answer1.strip().lower() == answer2.strip().lower()

def validate_rewritten_sample(
    sample: Dict,
    original_question: str,
    golden_answer: str,
    answer_match_mode: str = "strict",
    verbose: bool = False
) -> tuple[bool, str]:
    """
    Validate the quality of rewritten sample.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        if 'messages' not in sample or len(sample['messages']) < 2:
            return False, "Missing messages field or insufficient messages"
        
        content = sample['messages'][1]['content']
        
        # 1. Check answer exists and matches
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if not answer_match:
            return False, "Missing <answer> tag"
        
        extracted_answer = answer_match.group(1).strip()
        
        # Check answer match
        # NOTE: For EM evaluation alignment, default is strict (exact match after strip, case-insensitive).
        # Use --answer_match_mode substring only for debugging / higher yield.
        if answer_match_mode == "strict":
            if not em_match(extracted_answer, golden_answer):
                return False, f"Answer mismatch(strict): expected '{golden_answer}', got '{extracted_answer}'"
        elif answer_match_mode == "substring":
            if not em_match(extracted_answer, golden_answer):
                if golden_answer.lower() not in extracted_answer.lower() and extracted_answer.lower() not in golden_answer.lower():
                    return False, f"Answer mismatch(substring): expected '{golden_answer}', got '{extracted_answer}'"
        else:
            return False, f"Unknown answer_match_mode: {answer_match_mode}"
        
        # 2. Check reasoning exists (try multiple formats)
        # Note: SYSTEM_PROMPT uses `<think>` but model might generate various formats
        if not re.search(r'<think>', content, re.IGNORECASE) and not re.search(r'`<think>`', content, re.IGNORECASE):
            return False, "Missing reasoning tag (expected <think> or <think>)"
        
        # 3. Check action and feedback consistency
        has_recall = '<recall>' in content
        has_search = '<search>' in content
        has_memory = '<memory>' in content
        has_information = '<information>' in content
        
        if not (has_recall or has_search):
            return False, "Missing action tags (<recall> or <search>)"
        
        # Check tag pairing rules
        if has_recall and not has_memory:
            return False, "Has <recall> but missing <memory> (must pair recall with memory)"
        if has_search and not has_information:
            return False, "Has <search> but missing <information> (must pair search with information)"
        
        # Check for incorrect pairings
        if has_recall and has_information:
            # Check if information is used with recall (incorrect pairing)
            recall_positions = [m.start() for m in re.finditer(r'<recall>', content)]
            information_positions = [m.start() for m in re.finditer(r'<information>', content)]
            # If information appears after a recall, it's a pairing error
            for recall_pos in recall_positions:
                for info_pos in information_positions:
                    if info_pos > recall_pos:
                        # Check if there's a search between them
                        search_between = re.search(r'<search>', content[recall_pos:info_pos])
                        if not search_between:
                            return False, "Incorrect pairing: <recall> should use <memory>, not <information>"
        
        if has_search and has_memory:
            # Check if memory is used with search (incorrect pairing)
            search_positions = [m.start() for m in re.finditer(r'<search>', content)]
            memory_positions = [m.start() for m in re.finditer(r'<memory>', content)]
            # If memory appears after a search, it's a pairing error
            for search_pos in search_positions:
                for mem_pos in memory_positions:
                    if mem_pos > search_pos:
                        # Check if there's a recall between them
                        recall_between = re.search(r'<recall>', content[search_pos:mem_pos])
                        if not recall_between:
                            return False, "Incorrect pairing: <search> should use <information>, not <memory>"
        
        # 4. Check answer length (warning but not fail)
        if len(extracted_answer) > 100:
            if verbose:
                print(f"[Warning] Answer is long ({len(extracted_answer)} chars): {extracted_answer[:50]}...")
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation exception: {e}"

# ==============================================================================
# Rewrite Functions
# ==============================================================================

def safe_json_loads(raw_text: str) -> tuple[Optional[Dict], Optional[str]]:
    """
    Robust JSON loader for cases where the model returns extra text around a JSON object.
    Returns (obj, error_message).
    """
    # Fast path
    try:
        return json.loads(raw_text), None
    except Exception:
        pass

    # Strip markdown fences
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0].strip()

    # Try parse again
    try:
        return json.loads(text), None
    except Exception:
        pass

    # Fallback: extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"Failed to parse extracted JSON candidate: {e}"

    return None, "No JSON object found"

def is_nq_easy_recall_candidate(question: str) -> bool:
    """
    Heuristic: identify NQ questions that are likely common knowledge / definitional,
    so forcing <recall> won't create too much policy mismatch.
    """
    q = question.strip().lower()
    # Short definitional / concept questions
    easy_prefixes = (
        "what is ",
        "what are ",
        "define ",
        "meaning of ",
        "definition of ",
        "capital of ",
        "where is ",
        "where are ",
    )
    if q.startswith(easy_prefixes):
        return True

    # Avoid forcing recall on clearly lookup-heavy patterns
    hard_markers = (
        "who played",
        "voice of",
        "episodes",
        "season",
        "released",
        "release date",
        "born",
        "birth",
        "produced",
        "producer",
        "directed",
        "director",
        "cast",
        "box office",
        "population",
        "how many",
        "when did",
        "when was",
        "in 19",
        "in 20",
    )
    if any(m in q for m in hard_markers):
        return False

    # Very short questions are often common-ish
    if len(q) <= 40 and q.endswith("?"):
        return True

    return False

def rewrite_single_sample(
    key_manager: APIKeyManager,
    question: str,
    golden_answer: str,
    data_source: str = 'nq',
    model: str = "gpt-4",
    answer_match_mode: str = "strict",
    desired_action: Optional[str] = None,
) -> Optional[Dict]:
    """Rewrite a single (question, golden_answer) pair into AutoSearch format"""
    
    system_prompt = get_system_prompt(data_source)
    user_prompt = make_user_prompt(question, golden_answer, data_source, desired_action=desired_action)
    
    for attempt in range(3):  # Retry up to 3 times
        try:
            client = key_manager.get_client()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7 + (attempt * 0.1)  # Slightly increase temp on retry
            )
            
            raw_content = response.choices[0].message.content
            
            result, json_err = safe_json_loads(raw_content)
            if result is None:
                print(f"[JSON Error] Attempt {attempt+1}: {json_err}")
                print(f"[JSON Error] Raw content (first 200 chars): {raw_content[:200]}...")
                continue
            
            is_valid, error_msg = validate_rewritten_sample(
                result,
                question,
                golden_answer,
                answer_match_mode=answer_match_mode,
                verbose=(attempt == 2),
            )
            if is_valid:
                # Add metadata
                result['meta'] = {
                    'source': 'r1_rewritten',
                    'original_question': question,
                    'golden_answer': golden_answer,
                    'data_source': data_source,
                    'model': model,
                    'desired_action': desired_action,
                }
                return result
            else:
                if attempt == 2:  # Only print error on final attempt
                    print(f"[Validation Failed] Attempt {attempt+1} for question: {question[:50]}...")
                    print(f"  Error: {error_msg}")
                
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
                    data_source = data.get('data_source', 'nq').strip()
                    
                    if question and golden_answer:
                        samples.append({
                            'question': question,
                            'golden_answer': golden_answer,
                            'data_source': data_source
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
    parser.add_argument(
        "--answer_match_mode",
        type=str,
        default="strict",
        choices=["strict", "substring"],
        help="Answer matching mode for validation. strict aligns with EM (default). substring is more lenient."
    )
    parser.add_argument(
        "--nq_target_recall_ratio",
        type=float,
        default=0.2,
        help="Target recall ratio for NQ samples only (default: 0.2). Uses heuristics + forcing to reach the ratio."
    )
    
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
    
    # Decide desired_action for NQ samples to increase recall ratio
    nq_indices = [i for i, s in enumerate(samples) if s.get('data_source') == 'nq']
    easy_nq_indices = [i for i in nq_indices if is_nq_easy_recall_candidate(samples[i]['question'])]
    target_nq_recall = int(round(len(nq_indices) * float(args.nq_target_recall_ratio))) if nq_indices else 0
    target_nq_recall = max(0, min(target_nq_recall, len(easy_nq_indices)))
    forced_recall_set = set(random.sample(easy_nq_indices, k=target_nq_recall)) if target_nq_recall > 0 else set()

    for i, s in enumerate(samples):
        if s.get('data_source') == 'nq':
            s['desired_action'] = 'recall' if i in forced_recall_set else None
        else:
            s['desired_action'] = None

    if nq_indices:
        print(f"\nNQ action routing:")
        print(f"  NQ total: {len(nq_indices)}")
        print(f"  NQ easy candidates: {len(easy_nq_indices)}")
        print(f"  NQ forced recall: {len(forced_recall_set)} (target_ratio={args.nq_target_recall_ratio})")

    # Show data source distribution
    data_source_counts = {}
    for sample in samples:
        ds = sample.get('data_source', 'nq')
        data_source_counts[ds] = data_source_counts.get(ds, 0) + 1
    print(f"\nData source distribution:")
    for ds, count in data_source_counts.items():
        print(f"  {ds}: {count} samples")
    
    print(f"\nTotal samples to rewrite: {len(samples)}")
    print(f"Using {args.workers} workers")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    
    # Concurrent rewriting with periodic saving
    results = []
    results_lock = threading.Lock()
    save_interval = 100  # Save every 100 samples
    
    def rewrite_wrapper(sample):
        return rewrite_single_sample(
            key_manager,
            sample['question'],
            sample['golden_answer'],
            sample.get('data_source', 'nq'),
            args.model,
            answer_match_mode=args.answer_match_mode,
            desired_action=sample.get('desired_action'),
        )
    
    def save_results(results_list, output_path, is_final=False):
        """Save results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if not is_final:
            print(f"\n[Progress] Saved {len(results_list)} samples to {output_path}")
    
    print("\nStarting rewrite process...")
    if len(samples) >= save_interval:
        print(f"[Info] Results will be saved every {save_interval} samples and at the end")
    else:
        print(f"[Info] Results will be saved at the end")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(rewrite_wrapper, sample) for sample in samples]
        
        completed_count = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(samples), desc="Rewriting"):
            try:
                res = future.result()
                if res:
                    with results_lock:
                        results.append(res)
                        completed_count = len(results)
                        
                        # Periodic save
                        if completed_count % save_interval == 0:
                            save_results(results, args.output, is_final=False)
            except Exception as e:
                print(f"\n[Error] Exception in rewrite: {e}")
    
    # Final save
    save_results(results, args.output, is_final=True)
    
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
        
        # Show data source distribution in results
        result_sources = {}
        for result in results:
            ds = result.get('meta', {}).get('data_source', 'unknown')
            result_sources[ds] = result_sources.get(ds, 0) + 1
        if result_sources:
            print(f"\nResult data source distribution:")
            for ds, count in result_sources.items():
                print(f"  {ds}: {count} samples ({count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
