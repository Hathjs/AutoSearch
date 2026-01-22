#!/usr/bin/env python3
"""
Simple evaluation script for AutoSearch V2 SFT model.
Tests the model's ability to generate correct XML tags and answers.
Supports multi-turn interaction with search and recall actions.
"""

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import re
from typing import List, Dict, Tuple, Optional
import json
import requests
import signal
import sys
import threading
import time
import os

# Add project root to path to import recall_tool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from search_r1.search.recall_tool import recall_internal_knowledge

def check_cuda_status():
    """Check CUDA device status and memory usage"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    try:
        # Try to synchronize - if this hangs, CUDA operations are blocked
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"CUDA OK - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    except Exception as e:
        return f"CUDA ERROR: {e}"


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


class StopOnSequence(StoppingCriteria):
    """Stop generation when encountering specific sequences like </search>, </recall>, </answer>"""
    def __init__(self, target_sequences: List[str], tokenizer):
        self.target_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in target_sequences]
        # Filter out empty sequences
        self.target_ids = [tid for tid in self.target_ids if len(tid) > 0]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer
        
        if len(self.target_lengths) == 0:
            raise ValueError("No valid target sequences provided")

    def __call__(self, input_ids, scores, **kwargs):
        # Safety checks
        try:
            if input_ids.shape[1] == 0:
                return False
            
            if len(self.target_lengths) == 0:
                return False
            
            min_length = min(self.target_lengths)
            if input_ids.shape[1] < min_length:
                return False
            
            # Use CPU for comparison to avoid GPU numerical issues
            # Only check the last few tokens to avoid memory issues
            # Check up to the maximum target length + some buffer
            max_check_length = max(self.target_lengths) + 10
            check_length = min(max_check_length, input_ids.shape[1])
            input_ids_cpu = input_ids[0, -check_length:].cpu()
            
            for i, target_id in enumerate(self.target_ids):
                if len(target_id) == 0:
                    continue
                if input_ids_cpu.shape[0] < len(target_id):
                    continue
                
                # Compare using CPU tensors with proper dtype
                target_tensor = torch.tensor(target_id, dtype=input_ids.dtype)
                # Compare the last len(target_tensor) tokens
                if torch.equal(input_ids_cpu[-len(target_tensor):], target_tensor):
                    return True
        except Exception as e:
            # If comparison fails, don't stop (fail-safe)
            # Don't print here to avoid flooding output during generation
            return False
        
        return False


def search_query(query: str, search_url: str = "http://127.0.0.1:8000/retrieve", topk: int = 3) -> str:
    """
    Call the retrieval server to search for a query.
    
    Args:
        query: Search query string
        search_url: URL of the retrieval server
        topk: Number of documents to retrieve
        
    Returns:
        Formatted search results string
    """
    try:
        payload = {
            "queries": [query],
            "topk": topk,
            "return_scores": True
        }
        response = requests.post(search_url, json=payload, timeout=10)
        response.raise_for_status()
        results = response.json()['result']
        
        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference
        
        return _passages2string(results[0])
    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Search API call failed: {e}")
        return "Search service unavailable. Please check if retrieval server is running."
    except Exception as e:
        print(f"[WARNING] Error processing search results: {e}")
        return "Error processing search results."


def parse_response(text: str) -> Dict[str, any]:
    """Parse model response and extract XML tags."""
    result = {
        'has_reasoning': False,
        'has_recall': False,
        'has_search': False,
        'has_memory': False,
        'has_information': False,
        'has_answer': False,
        'answer': None,
        'recall_queries': [],
        'search_queries': [],
        'raw_text': text
    }
    
    # Check for reasoning
    if re.search(r'<think>.*?</think>', text, re.DOTALL):
        result['has_reasoning'] = True
    
    # Check for recall
    recall_matches = re.findall(r'<recall>(.*?)</recall>', text, re.DOTALL)
    if recall_matches:
        result['has_recall'] = True
        result['recall_queries'] = [q.strip() for q in recall_matches]
    
    # Check for memory
    if re.search(r'<memory>.*?</memory>', text, re.DOTALL):
        result['has_memory'] = True
    
    # Check for search
    search_matches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
    if search_matches:
        result['has_search'] = True
        result['search_queries'] = [q.strip() for q in search_matches]
    
    # Check for information
    if re.search(r'<information>.*?</information>', text, re.DOTALL):
        result['has_information'] = True
    
    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        result['has_answer'] = True
        result['answer'] = answer_match.group(1).strip()
    
    return result


def evaluate_model(
    model_path: str,
    test_questions: List[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    device: str = "cuda",
    search_url: Optional[str] = "http://127.0.0.1:8000/retrieve",
    max_turns: int = 10,
    enable_search: bool = True
):
    """
    Evaluate the SFT model on test questions with multi-turn interaction.
    
    Args:
        model_path: Path to the fine-tuned model
        test_questions: List of test questions
        max_new_tokens: Maximum new tokens per turn
        temperature: Sampling temperature
        device: Device to use (cuda/cpu)
        search_url: URL of the retrieval server (None to disable search)
        max_turns: Maximum number of interaction turns
        enable_search: Whether to enable search API calls
    """
    
    print(f"Loading model from {model_path}...")
    
    # Convert to absolute path and check if it exists
    import os
    model_path_abs = os.path.abspath(os.path.expanduser(model_path))
    
    if not os.path.exists(model_path_abs):
        raise ValueError(f"Model path does not exist: {model_path_abs}")
    
    if not os.path.isdir(model_path_abs):
        raise ValueError(f"Model path is not a directory: {model_path_abs}")
    
    # Load tokenizer (local files only, disable HuggingFace Hub)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_abs, 
        trust_remote_code=True,
        local_files_only=True  # Disable HuggingFace Hub, use local files only
    )
    
    # Disable Flash Attention to avoid hangs (use eager implementation instead)
    # Flash Attention requires specific dtype/device configuration and may cause hangs
    print("  Loading with eager attention (Flash Attention disabled)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path_abs,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager",  # CRITICAL: Disable Flash Attention to avoid hangs
        local_files_only=True  # Disable HuggingFace Hub, use local files only
    )
    if device == "cpu":
        model = model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Clear default generation config to avoid conflicts with our explicit parameters
    # Create a fresh GenerationConfig to override defaults
    try:
        from transformers import GenerationConfig
        # Create a minimal config that will be overridden by gen_kwargs
        model.generation_config = GenerationConfig(
            do_sample=False,  # Default to greedy decoding
            temperature=None,
            top_p=None,
            top_k=None,
        )
        print(f"[INFO] Reset generation_config to minimal config")
    except Exception as e:
        print(f"[WARNING] Could not reset generation_config: {e}")
        # Fallback: try to clear individual attributes
        if hasattr(model, 'generation_config'):
            try:
                if hasattr(model.generation_config, 'temperature'):
                    model.generation_config.temperature = None
                if hasattr(model.generation_config, 'top_p'):
                    model.generation_config.top_p = None
                if hasattr(model.generation_config, 'top_k'):
                    model.generation_config.top_k = None
                if hasattr(model.generation_config, 'do_sample'):
                    model.generation_config.do_sample = False
            except Exception as e2:
                print(f"[WARNING] Could not clear generation_config attributes: {e2}")
    
    # Get EOS token IDs (for Qwen2.5 and other models)
    curr_eos = [tokenizer.eos_token_id]
    if hasattr(tokenizer, 'im_end_id') and tokenizer.im_end_id is not None:
        curr_eos.append(tokenizer.im_end_id)
    
    # Stopping criteria for detecting action tags
    # Use simpler sequences to avoid tokenization issues
    target_sequences = ["</search>", "</recall>", "</answer>"]
    try:
        stopping_criteria = StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    except Exception as e:
        print(f"[WARNING] Failed to create stopping criteria: {e}. Continuing without it.")
        stopping_criteria = None
    
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"\n{'='*80}")
        print(f"Test {i+1}/{len(test_questions)}: {question}")
        print(f"{'='*80}")
        
        # Generate initial prompt
        initial_prompt = make_prefix(question)
        
        # Apply chat template (consistent with training)
        # Training uses chat_template, so evaluation should too
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": initial_prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = initial_prompt
        
        # Multi-turn interaction loop
        full_response = ""
        turn = 0
        
        # Flag for graceful shutdown
        stop_generation = threading.Event()
        
        def signal_handler(sig, frame):
            """Handle Ctrl+C gracefully"""
            print("\n\n[Received interrupt signal (Ctrl+C). Stopping generation...]")
            stop_generation.set()
            sys.exit(0)
        
        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)
        
        print(f"\n[Starting multi-turn interaction (max {max_turns} turns)]")
        print(f"[Press Ctrl+C to stop gracefully]")
        
        try:
            while turn < max_turns:
                turn += 1
                print(f"\n--- Turn {turn} ---")
            
                # Check if stop was requested
                if stop_generation.is_set():
                    print("\n[Stop requested, breaking loop]")
                    break
                
                # Tokenize current prompt
                print(f"[Tokenizing prompt... (length: {len(prompt)} chars)]")
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                attention_mask = torch.ones_like(input_ids)
                print(f"[Tokenized to {input_ids.shape[1]} tokens]")
                
                # Generate with stopping criteria
                print(f"[Generating (max_new_tokens={max_new_tokens})... This may take 10-60 seconds...]")
                cuda_status = check_cuda_status()
                print(f"[CUDA Status: {cuda_status}]")
                print(f"[GPU Memory Check: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved]")
                sys.stdout.flush()  # Force flush output
                
                start_time = time.time()
                
                try:
                    # Prepare generation kwargs
                    gen_kwargs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'max_new_tokens': max_new_tokens,
                        'pad_token_id': tokenizer.pad_token_id,
                        'eos_token_id': tokenizer.eos_token_id,
                    }
                    
                    # Add stopping criteria if available
                    # Temporarily disable to debug hanging issues
                    # TODO: Re-enable after fixing stopping_criteria
                    # if stopping_criteria is not None:
                    #     gen_kwargs['stopping_criteria'] = stopping_criteria
                    
                    # Handle sampling parameters safely
                    if temperature > 0:
                        gen_kwargs['do_sample'] = True
                        gen_kwargs['temperature'] = max(0.01, min(temperature, 2.0))  # Clamp temperature
                    else:
                        gen_kwargs['do_sample'] = False
                        # For greedy decoding, explicitly unset sampling parameters
                        # This ensures they don't interfere with do_sample=False
                        if 'temperature' in gen_kwargs:
                            del gen_kwargs['temperature']
                        if 'top_p' in gen_kwargs:
                            del gen_kwargs['top_p']
                        if 'top_k' in gen_kwargs:
                            del gen_kwargs['top_k']
                    
                    # Debug: print generation config
                    print(f"[DEBUG] Generation kwargs: do_sample={gen_kwargs.get('do_sample', 'N/A')}, "
                          f"temperature={gen_kwargs.get('temperature', 'N/A')}, "
                          f"max_new_tokens={gen_kwargs.get('max_new_tokens', 'N/A')}")
                    sys.stdout.flush()
                    
                    # Direct generation call with better error handling
                    # Note: If this hangs, it's likely a CUDA operation issue
                    # Run `python scripts/check_cuda.py` in another terminal to diagnose
                    print(f"[DEBUG] Starting model.generate() call...")
                    print(f"[DEBUG] If this hangs, run 'python scripts/check_cuda.py' in another terminal to diagnose CUDA")
                    sys.stdout.flush()
                    
                    try:
                        with torch.no_grad():
                            outputs = model.generate(**gen_kwargs)
                        print(f"[DEBUG] model.generate() returned successfully")
                        sys.stdout.flush()
                    except Exception as gen_error:
                        elapsed = time.time() - start_time
                        print(f"[ERROR] Generation failed after {elapsed:.1f}s: {gen_error}")
                        print(f"[DEBUG] CUDA Status: {check_cuda_status()}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    elapsed = time.time() - start_time
                    print(f"[Generation completed in {elapsed:.1f}s. Output length: {outputs.shape[1]} tokens]")
                except KeyboardInterrupt:
                    print("\n[Generation interrupted by user]")
                    raise
                except RuntimeError as e:
                    error_str = str(e)
                    if "CUDA" in error_str or "cuda" in error_str or "out of memory" in error_str.lower():
                        print(f"[CUDA ERROR during generation: {error_str}]")
                        print("[Trying to clear CUDA cache...]")
                        torch.cuda.empty_cache()
                        break
                    else:
                        print(f"[RuntimeError during generation: {error_str}]")
                        import traceback
                        traceback.print_exc()
                        break
                except Exception as e:
                    print(f"[ERROR during generation: {e}]")
                    import traceback
                    traceback.print_exc()
                    break
                
                # Check if generation ended with EOS
                if outputs[0][-1].item() in curr_eos:
                    generated_tokens = outputs[0][input_ids.shape[1]:]
                    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
                    full_response += output_text
                    print(f"Generated: {output_text[:200]}...")
                    print("[EOS reached, ending interaction]")
                    break
                
                # Decode new tokens
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
                full_response += output_text
                
                print(f"Generated: {output_text[:200]}...")
                
                # Check what action was taken
                full_text_so_far = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                search_match = re.search(r'<search>(.*?)</search>', full_text_so_far, re.DOTALL)
                recall_match = re.search(r'<recall>(.*?)</recall>', full_text_so_far, re.DOTALL)
                answer_match = re.search(r'<answer>(.*?)</answer>', full_text_so_far, re.DOTALL)
                
                # Handle actions
                if answer_match:
                    # Final answer found
                    print("[Final answer detected, ending interaction]")
                    break
                elif search_match:
                    # Model requested search
                    search_query_text = search_match.group(1).strip()
                    print(f"[Model requested search: '{search_query_text}']")
                    
                    if enable_search and search_url:
                        try:
                            search_results = search_query(search_query_text, search_url)
                            print(f"[Search results retrieved]")
                            # Append search feedback to prompt
                            prompt += output_text + f"\n\n<information>{search_results}</information>\n\n"
                        except Exception as e:
                            print(f"[WARNING] Search failed: {e}")
                            prompt += output_text + "\n\n<information>Search service unavailable.</information>\n\n"
                    else:
                        print("[Search disabled in evaluation mode]")
                        prompt += output_text + "\n\n<information>Search disabled in evaluation mode.</information>\n\n"
                elif recall_match:
                    # Model requested recall (internal memory)
                    recall_query_text = recall_match.group(1).strip()
                    print(f"[Model requested recall: '{recall_query_text}']")
                    
                    # Call recall tool to guide model to recall knowledge
                    try:
                        print(f"[Calling recall tool...]")
                        memory_content = recall_internal_knowledge(
                            query=recall_query_text,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            max_new_tokens=256,
                            temperature=0.1  # Low temperature for deterministic recall
                        )
                        print(f"[Memory recalled: {memory_content[:150]}...]")
                        # Append recall output and memory feedback to prompt
                        prompt += output_text + f"\n\n{memory_content}\n\n"
                    except Exception as e:
                        print(f"[WARNING] Recall tool failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to placeholder
                        prompt += output_text + "\n\n<memory>No record found.</memory>\n\n"
                else:
                    # No action tag found, continue generation
                    if turn >= max_turns:
                        print(f"[Max turns ({max_turns}) reached, stopping]")
                        break
                    # Continue to next turn
                    prompt += output_text
        
        except KeyboardInterrupt:
            print("\n\n[Interrupted by user. Saving partial results...]")
        except Exception as e:
            print(f"\n\n[ERROR: {e}]")
            import traceback
            traceback.print_exc()
        
        # Parse final response
        parsed = parse_response(full_response)
        parsed['question'] = question
        parsed['num_turns'] = turn
        
        # Print results
        print(f"\n{'='*80}")
        print("Generated Response:")
        print(f"{full_response[:1000]}..." if len(full_response) > 1000 else full_response)
        print(f"\n{'='*80}")
        print("Parsed Results:")
        print(f"  - Num Turns: {parsed['num_turns']}")
        print(f"  - Has Reasoning: {parsed['has_reasoning']}")
        print(f"  - Has Recall: {parsed['has_recall']} (queries: {parsed['recall_queries']})")
        print(f"  - Has Memory: {parsed['has_memory']}")
        print(f"  - Has Search: {parsed['has_search']} (queries: {parsed['search_queries']})")
        print(f"  - Has Information: {parsed['has_information']}")
        print(f"  - Has Answer: {parsed['has_answer']}")
        if parsed['answer']:
            print(f"  - Answer: {parsed['answer']}")
        
        results.append(parsed)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    total = len(results)
    print(f"Total questions: {total}")
    print(f"Has Reasoning: {sum(1 for r in results if r['has_reasoning'])}/{total} ({sum(1 for r in results if r['has_reasoning'])/total*100:.1f}%)")
    print(f"Has Recall: {sum(1 for r in results if r['has_recall'])}/{total} ({sum(1 for r in results if r['has_recall'])/total*100:.1f}%)")
    print(f"Has Search: {sum(1 for r in results if r['has_search'])}/{total} ({sum(1 for r in results if r['has_search'])/total*100:.1f}%)")
    print(f"Has Answer: {sum(1 for r in results if r['has_answer'])}/{total} ({sum(1 for r in results if r['has_answer'])/total*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AutoSearch V2 SFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="JSONL file with test questions (one per line, JSON format with 'question' field)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        default=None,
        help="Test questions as command-line arguments"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per turn (default: 512, reduce if generation is slow)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--search_url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="URL of the retrieval server (set to empty string to disable search)"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum number of interaction turns"
    )
    parser.add_argument(
        "--disable_search",
        action="store_true",
        help="Disable search API calls (for testing without retrieval server)"
    )
    
    args = parser.parse_args()
    
    # Load test questions
    test_questions = []
    
    if args.test_file:
        with open(args.test_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'question' in data:
                        test_questions.append(data['question'])
                    elif 'messages' in data:
                        # Extract question from messages
                        for msg in data['messages']:
                            if msg['role'] == 'user':
                                test_questions.append(msg['content'])
                                break
    
    if args.questions:
        test_questions.extend(args.questions)
    
    if not test_questions:
        # Default test questions
        test_questions = [
            "What is the capital of France?",
            "Who wrote the novel '1984'?",
            "What is the current population of Tokyo?",
            "What is the chemical formula for water?",
            "Who won the Nobel Prize in Physics in 2023?",
        ]
        print("No test questions provided, using default questions...")
    
    # Check if search URL is provided
    search_url = args.search_url if args.search_url else None
    enable_search = not args.disable_search and search_url is not None
    
    if enable_search:
        print(f"\n[INFO] Search enabled. Retrieval server URL: {search_url}")
        # Test connection
        try:
            test_response = requests.get(search_url.replace("/retrieve", "/health") if "/retrieve" in search_url else search_url, timeout=2)
            print(f"[INFO] Retrieval server connection OK")
        except:
            print(f"[WARNING] Cannot connect to retrieval server. Search calls may fail.")
            print(f"[INFO] You can start the retrieval server with: bash retrieval_launch.sh")
    else:
        print(f"\n[INFO] Search disabled. Model will receive placeholder responses for <search> actions.")
    
    # Evaluate
    results = evaluate_model(
        model_path=args.model_path,
        test_questions=test_questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        search_url=search_url,
        max_turns=args.max_turns,
        enable_search=enable_search
    )
    
    # Save results
    output_file = "eval_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")
