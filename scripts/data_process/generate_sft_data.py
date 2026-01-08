import os
import json
import yaml
import time
import random
import re
import argparse
import concurrent.futures
import httpx # Added httpx
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Optional

# ==============================================================================
# 1. Configuration & API Key Management
# ==============================================================================

class APIKeyManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.keys = self.config['api_key_list']
        self.current_index = 0
        print(f"Loaded {len(self.keys)} API keys.")

    def get_client(self) -> OpenAI:
        """Round-robin rotation of API keys with SSL verification disabled."""
        if not self.keys:
            raise ValueError("No API keys found in configuration")
            
        key_info = self.keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.keys)
        
        # Create a custom http client that disables SSL verification
        # This is useful for corporate networks or VPNs with self-signed certs
        http_client = httpx.Client(verify=False)
        
        return OpenAI(
            api_key=key_info['api_key'],
            base_url=key_info['base_url'],
            http_client=http_client
        )

# ==============================================================================
# 2. Prompt Templates
# ==============================================================================

SYSTEM_PROMPT = """You are an expert data generator for 'AutoSearch V2'.
Your task is to generate high-quality SFT training data in JSON format.
The agent follows a strict XML protocol:
1. <think>...reasoning...</think>
2. <recall>query</recall> OR <search>query</search>
3. System Feedback: <memory>...content...</memory> OR <information>...content...</information>
4. <answer>Final Answer</answer>

**CRITICAL CONSTRAINT FOR <answer>:**
- The content inside <answer>...</answer> MUST be a **SHORT ENTITY** or **EXACT PHRASE** (e.g., "Paris", "$150.00", "Michael Scott", "1976").
- DO NOT write complete sentences inside <answer>.
- DO NOT include punctuation like periods at the end unless part of the entity.
- Example:
  - BAD: <answer>The capital is Paris.</answer>
  - GOOD: <answer>Paris</answer>

**DIVERSITY REQUIREMENT:**
- Do NOT generate generic questions like "Capital of France" or "Apple Stock".
- Be CREATIVE. Use specific entities, dates, and less common examples within the requested topic.

Output MUST be a single valid JSON object with the format:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

PROMPT_TYPE_A = """Generate a 'Type A (Internal Recall)' sample.
- User asks a static fact (History, Science, Geography).
- Assistant MUST use <recall>query</recall>.
- Assistant SIMULATES the feedback: <memory>Exact fact...</memory>.
- Assistant generates <answer>SHORT_ENTITY</answer>.
- NO <search> allowed.
"""

PROMPT_TYPE_B = """Generate a 'Type B (External Search)' sample.
- User asks a dynamic/real-time question (Stock price, News, Recent Event, Weather).
- Assistant MUST use <search>query</search>.
- Assistant SIMULATES the feedback: <information>Search results...</information>.
- Assistant generates <answer>SHORT_VALUE_OR_ENTITY</answer>.
- NO <recall> allowed.
"""

PROMPT_TYPE_C = """Generate a 'Type C (Trap/Hallucination Check)' sample.
- User asks a question with a FALSE PREMISE (e.g., 'Who is the CEO of Apple in 1800?').
- Assistant flow:
  1. <think>Suspicious premise...</think>
  2. <recall>Check entity</recall>
  3. Feedback: <memory>No record found.</memory>
  4. <think>Pivot logic: No record implies false premise.</think>
  5. 5. <answer>N/A</answer> (ALWAYS use 'N/A' for refusal)."
"""

# ==============================================================================
# 3. Generator Logic
# ==============================================================================

# Topic Pools for Diversity
TOPICS_A = [
    "Chemistry Elements", "Quantum Physics", "Ancient Rome", "World Geography", "Nobel Prize Winners",
    "Biological Taxonomy", "Famous Painters", "Classical Music Composers", "Human Anatomy", "Solar System",
    "World War II Battles", "Chemical Formulas", "Capital Cities", "US Presidents", "Mathematical Constants",
    "Famous Novels", "Architectural Wonders", "Olymptic History", "Philosophy Concepts", "Geological Eras"
]

TOPICS_B = [
    "Stock Prices (Tech)", "Stock Prices (Energy)", "Recent Sports Scores (NBA/NFL/Soccer)", "Weather in Major Cities",
    "Upcoming Movie Releases", "Cryptocurrency Prices", "Recent Election Results", "Technological Breakthroughs 2024",
    "Viral Internet Trends", "New Video Game Releases", "Award Show Winners 2024", "Global Population Stats",
    "Current Exchange Rates", "Flight Status", "Concert Tour Dates", "Latest Smartphone Specs",
    "SpaceX Launch Schedule", "Climate Change Data 2024", "Best Selling Books This Week"
]

TOPICS_C = [
    "Historical Anachronisms (e.g., iPhone in 1800)", "Fictional Events treated as real", "Future dates in the past",
    "Non-existent Movie Sequels", "Celebrities born in wrong centuries", "Inventions attributed to wrong people",
    "Geographical Impossibilities (e.g., Ocean in Swiss)", "Biological Hybrids (e.g., Flying Elephants)",
    "Fake Political Events", "Made-up Chemical Elements", "Sports teams in wrong leagues", "Dead people doing recent things"
]

def validate_sample(sample: Dict, sample_type: str) -> bool:
    """Basic validation of XML tags."""
    try:
        content = sample['messages'][1]['content']
        if '<think>' not in content or '<answer>' not in content:
            return False
        
        if sample_type == 'A':
            return '<recall>' in content and '<memory>' in content
        elif sample_type == 'B':
            return '<search>' in content and '<information>' in content
        elif sample_type == 'C':
            return '<recall>' in content and '<memory>' in content and 'No record' in content
            
    except Exception:
        return False
    return True

def generate_single_sample(key_manager: APIKeyManager, sample_type: str, model: str) -> Optional[Dict]:
    """Generates one valid sample with retry logic."""
    
    if sample_type == 'A':
        topic = random.choice(TOPICS_A)
        user_prompt = PROMPT_TYPE_A + f"\n\n**IMPORTANT REQUIREMENT:** The question MUST be about **{topic}**. Make it specific and diverse."
    elif sample_type == 'B':
        topic = random.choice(TOPICS_B)
        user_prompt = PROMPT_TYPE_B + f"\n\n**IMPORTANT REQUIREMENT:** The question MUST be about **{topic}**. Make it specific and diverse."
    else:
        topic = random.choice(TOPICS_C)
        user_prompt = PROMPT_TYPE_C + f"\n\n**IMPORTANT REQUIREMENT:** Create a tricky question related to **{topic}**."

    for attempt in range(3): # Retry up to 3 times
        try:
            # Get a client for this attempt (rotates keys)
            client = key_manager.get_client()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8 + (attempt * 0.1) # Increase temp slightly on retry
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
            except json.JSONDecodeError:
                print(f"[JSON Error] Raw content: {raw_content[:200]}...")
                continue
            
            if validate_sample(result, sample_type):
                # Tag metadata for debugging/filtering
                result['meta'] = {'type': sample_type, 'model': model}
                return result
            else:
                print(f"[Validation Failed] Type {sample_type}. Content: {str(result)[:200]}...")
                
        except Exception as e:
            print(f"[API Error] Attempt {attempt+1}: {e}") 
            # Simple backoff
            time.sleep(1)
            pass
            
    return None

# ==============================================================================
# 4. Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", type=str, default="data/autosearch_sft_raw.jsonl")
    parser.add_argument("--count_a", type=int, default=50)
    parser.add_argument("--count_b", type=int, default=50)
    parser.add_argument("--count_c", type=int, default=25)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4") 
    args = parser.parse_args()

    # 1. Load Config
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    try:
        key_manager = APIKeyManager(args.config)
    except Exception as e:
        print(f"Failed to load API keys: {e}")
        return

    # 2. Prepare Tasks
    tasks = []
    tasks.extend(['A'] * args.count_a)
    tasks.extend(['B'] * args.count_b)
    tasks.extend(['C'] * args.count_c)
    random.shuffle(tasks)

    print(f"Starting generation of {len(tasks)} samples with {args.workers} workers...")
    print(f"Model: {args.model}")
    print(f"Type Distribution: A={args.count_a}, B={args.count_b}, C={args.count_c}")
    
    results = []
    
    # 3. Concurrent Execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        # Note: key_manager is shared, but get_client() is lightweight
        futures = [executor.submit(generate_single_sample, key_manager, t, args.model) for t in tasks]
        
        # Process as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            res = future.result()
            if res:
                results.append(res)

    # 4. Save Results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone! Generated {len(results)}/{len(tasks)} valid samples.")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
