#!/usr/bin/env python3
"""
Simple evaluation script for AutoSearch V2 SFT model.
Tests the model's ability to generate correct XML tags and answers.
"""

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Dict, Tuple
import json

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
    device: str = "cuda"
):
    """Evaluate the SFT model on test questions."""
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"\n{'='*80}")
        print(f"Test {i+1}/{len(test_questions)}: {question}")
        print(f"{'='*80}")
        
        # Generate prompt
        prompt = make_prefix(question)
        
        # Apply chat template if available
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )
        
        # Parse response
        parsed = parse_response(generated_text)
        parsed['question'] = question
        
        # Print results
        print(f"\nGenerated Response:")
        print(f"{generated_text[:500]}..." if len(generated_text) > 500 else generated_text)
        print(f"\nParsed Results:")
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
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
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
    
    # Evaluate
    results = evaluate_model(
        model_path=args.model_path,
        test_questions=test_questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    # Save results
    output_file = "eval_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")
