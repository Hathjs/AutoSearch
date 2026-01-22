"""
Recall Tool for AutoSearch V2

This module implements the internal recall tool that guides the model
to recall knowledge from its parametric memory (training data).

The recall tool is designed to:
1. Decouple knowledge recall from logical reasoning
2. Guide the model to "remember" information it already knows
3. Support both evaluation and RL training scenarios
"""

import torch
import re
from typing import Optional


def recall_internal_knowledge(
    query: str,
    model,
    tokenizer,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    use_chat_template: bool = True
) -> str:
    """
    Recall tool: Guide the model to recall internal knowledge from its parametric memory.
    
    This function uses a specialized prompt to guide the model to recall what it knows
    about the given query. The model generates the memory content based on its training.
    
    Args:
        query: The recall query (extracted from <recall>query</recall>)
        model: The model instance (should be the same model being evaluated/trained)
        tokenizer: The tokenizer instance
        device: Device to use ("cuda" or "cpu")
        max_new_tokens: Maximum tokens to generate for memory content
        temperature: Sampling temperature (low temperature for more deterministic recall)
        use_chat_template: Whether to use chat template if available
        
    Returns:
        Memory content wrapped in <memory>...</memory> tags
    """
    # Specialized recall prompt to guide the model
    recall_prompt = f"""You are recalling internal knowledge from your training data.
Based on the following query, recall what you know about it from your parametric memory.
If you don't have this knowledge in your training data, respond with "No record found."

Query: {query}

Your knowledge:"""
    
    # Apply chat template if available and requested
    if use_chat_template and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": recall_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        formatted_prompt = recall_prompt
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(formatted_prompt, return_tensors='pt')
    if device == "cuda" and torch.cuda.is_available():
        input_ids = input_ids.to(device)
    else:
        device = "cpu"
        input_ids = input_ids.to(device)
    
    # Generate memory content
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Stop early if model generates </memory> tag
            stopping_criteria=None,  # We'll handle stopping manually if needed
        )
    
    # Decode the generated content
    generated_text = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # Clean up the generated text (remove any trailing tags or extra content)
    # If the model generated </memory> or other tags, extract only the content
    if '</memory>' in generated_text:
        generated_text = generated_text.split('</memory>')[0]
    if '<memory>' in generated_text:
        generated_text = generated_text.split('<memory>')[-1]
    
    # If empty or too short, provide a default response
    if not generated_text or len(generated_text.strip()) < 3:
        generated_text = "No record found."
    
    # Wrap in <memory> tags
    return f"<memory>{generated_text}</memory>"


def batch_recall_internal_knowledge(
    queries: list,
    model,
    tokenizer,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.1
) -> list:
    """
    Batch version of recall_internal_knowledge for processing multiple queries.
    
    Args:
        queries: List of recall queries
        model: The model instance
        tokenizer: The tokenizer instance
        device: Device to use
        max_new_tokens: Maximum tokens per query
        temperature: Sampling temperature
        
    Returns:
        List of memory contents wrapped in <memory>...</memory> tags
    """
    results = []
    for query in queries:
        memory = recall_internal_knowledge(
            query=query,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        results.append(memory)
    return results


def extract_recall_queries(text: str) -> list:
    """
    Extract all recall queries from text containing <recall>...</recall> tags.
    
    Args:
        text: Text containing recall tags
        
    Returns:
        List of recall queries (empty list if none found)
    """
    pattern = re.compile(r'<recall>(.*?)</recall>', re.DOTALL)
    matches = pattern.findall(text)
    return [match.strip() for match in matches]
