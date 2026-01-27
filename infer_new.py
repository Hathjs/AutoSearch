import transformers
import torch
import json
import re
import requests
import sys
import os
import argparse
from typing import List, Optional

# Add project root to path to import recall_tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_r1.search.recall_tool import recall_internal_knowledge

def make_prefix(question: str) -> str:
    """Generate system prompt for AutoSearch V2."""
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

Question: {question}
"""
    return prefix

class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text, action_type="search"):
    if action_type == "search":
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    elif action_type == "recall":
        pattern = re.compile(r"<recall>(.*?)</recall>", re.DOTALL)
    else:
        return None
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str, search_url: str):
    try:
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        response = requests.post(search_url, json=payload, timeout=10)
        results = response.json()['result']
                    
        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                if isinstance(doc_item, dict) and 'document' in doc_item:
                    # Handle return_scores=True format: {"document": {...}, "score": ...}
                    content = doc_item['document']['contents']
                elif 'contents' in doc_item:
                     # Handle return_scores=False format: {"contents": ..., ...}
                    content = doc_item['contents']
                else:
                    continue
                    
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference

        return _passages2string(results[0])
    except Exception as e:
        print(f"[WARNING] Search failed: {e}")
        return "Search service unavailable."

def compute_em_score(predicted: str, ground_truth: str) -> bool:
    if not predicted or not ground_truth:
        return False
    pred_norm = predicted.lower().strip()
    gt_norm = ground_truth.lower().strip()
    if pred_norm == gt_norm:
        return True
    if pred_norm in gt_norm or gt_norm in pred_norm:
        if len(pred_norm) >= 3 and len(gt_norm) >= 3:
            return True
    return False

def extract_answer(text: str) -> Optional[str]:
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="AutoSearch V2 Inference & Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_file", type=str, default=None, help="Path to test JSONL file")
    parser.add_argument("--question", type=str, default=None, help="Single question to test")
    parser.add_argument("--search_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--max_turns", type=int, default=10)
    args = parser.parse_args()

    # Load Model & Tokenizer
    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager" 
    )

    # Stopping Criteria
    target_sequences = ["</search>", " </search>", "</search>\n", 
                       "</recall>", " </recall>", "</recall>\n",
                       "</answer>", " </answer>", "</answer>\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    # EOS Tokens
    curr_eos = [tokenizer.eos_token_id]
    if hasattr(tokenizer, 'im_end_id') and tokenizer.im_end_id:
        curr_eos.append(tokenizer.im_end_id)
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id:
        curr_eos.append(tokenizer.pad_token_id)

    # Prepare Questions
    test_items = []
    if args.question:
        test_items.append({"question": args.question, "golden_answer": None})
    if args.test_file:
        with open(args.test_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                q = data.get('question')
                if not q and 'messages' in data:
                    for msg in data['messages']:
                        if msg['role'] == 'user':
                            q = msg['content']
                            break
                ga = data.get('meta', {}).get('golden_answer') or data.get('golden_answer') or data.get('answer')
                if q:
                    test_items.append({"question": q, "golden_answer": ga})

    print(f"Total questions to evaluate: {len(test_items)}")
    
    results = []
    
    for i, item in enumerate(test_items):
        question = item['question']
        golden_answer = item['golden_answer']
        
        print(f"\n{'='*80}")
        print(f"Test {i+1}/{len(test_items)}: {question}")
        print(f"{'='*80}")

        # Construct Prompt
        initial_prompt = make_prefix(question)
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": initial_prompt}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = initial_prompt

        full_response = ""
        turn = 0
        
        while turn < args.max_turns:
            turn += 1
            print(f"--- Turn {turn} ---")
            
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones_like(input_ids)
            
            try:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )
            except Exception as e:
                print(f"[Error] Generation failed: {e}")
                break

            # Decode only new tokens
            new_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=False) # Keep special chars to detect XML tags accurately if needed
            
            # Check for EOS
            is_eos = outputs[0][-1].item() in curr_eos
            
            # Append to history
            full_response += output_text
            print(f"Generated: {output_text.strip()}")
            
            # Check Actions
            # We check the newly generated text for completed tags
            # Note: infer.py checks full text, but checking new text is usually sufficient if we stop right after tag
            
            search_query = get_query(output_text, "search")
            recall_query = get_query(output_text, "recall")
            answer_match = extract_answer(output_text)
            
            if answer_match:
                print("[Final answer detected]")
                break
                
            if is_eos:
                print("[EOS detected]")
                break

            action_result = ""
            if search_query:
                print(f'[Searching: "{search_query}"]')
                search_res = search(search_query, args.search_url)
                action_result = f'\n\n<information>{search_res}</information>\n\n'
                print(f"[Information retrieved: {len(search_res)} chars]")
                
            elif recall_query:
                print(f'[Recalling: "{recall_query}"]')
                try:
                    memory_content = recall_internal_knowledge(
                        query=recall_query,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_new_tokens=256,
                        temperature=0.1
                    )
                    action_result = f'\n\n{memory_content}\n\n'
                    print(f"[Memory retrieved: {len(memory_content)} chars]")
                except Exception as e:
                    print(f"[Recall Error] {e}")
                    action_result = f'\n\n<memory>Error during recall.</memory>\n\n'
            
            if action_result:
                # Append result to prompt for next turn
                # Important: We must append the MODEL'S output + SYSTEM'S output
                # The 'prompt' variable currently holds history up to start of this turn.
                # We need to add what model just generated + the tool result.
                prompt += output_text + action_result
            else:
                # No action, just continue generation or stop if nothing happened
                # If model didn't output EOS and didn't output action, it might have been cut off by max_tokens
                prompt += output_text
                
        # Calculate Metrics
        final_answer = extract_answer(full_response)
        em_score = compute_em_score(final_answer, golden_answer) if golden_answer else None
        
        result_entry = {
            "question": question,
            "golden_answer": golden_answer,
            "model_answer": final_answer,
            "em_score": em_score,
            "num_turns": turn,
            "full_trace": full_response
        }
        results.append(result_entry)
        
        if golden_answer:
            print(f"Golden: {golden_answer}")
            print(f"Predicted: {final_answer}")
            print(f"EM Score: {'✓' if em_score else '✗'}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    valid_scores = [r['em_score'] for r in results if r['em_score'] is not None]
    if valid_scores:
        print(f"Accuracy: {sum(valid_scores)}/{len(valid_scores)} ({sum(valid_scores)/len(valid_scores)*100:.1f}%)")
    
    with open("eval_results_new.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved to eval_results_new.json")

if __name__ == "__main__":
    main()