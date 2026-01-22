import transformers
import torch
import random
from datasets import load_dataset
import requests
import sys
import os

# Add project root to path to import recall_tool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_r1.search.recall_tool import recall_internal_knowledge

question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = question.strip()
if question[-1] != '?':
    question += '?'
curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Prepare the message
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Define the custom stopping criterion
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
    import re
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

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria (support both search and recall)
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n",
                   "</recall>", " </recall>", "</recall>\n", " </recall>\n", "</recall>\n\n", " </recall>\n\n",
                   "</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

cnt = 0

if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(prompt)
# Encode the chat-formatted prompt and move it to the correct device
while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate text with the stopping criteria
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )

    if outputs[0][-1].item() in curr_eos:
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(output_text)
        break

    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check for search action
    search_query = get_query(full_text, action_type="search")
    recall_query = get_query(full_text, action_type="recall")
    
    if search_query:
        # External search
        print(f'[Searching: "{search_query}"]')
        search_results = search(search_query)
        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        print(search_text)
    elif recall_query:
        # Internal recall
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
            recall_text = f'\n\n{output_text}{memory_content}\n\n'
            prompt += recall_text
            cnt += 1
            print(recall_text)
        except Exception as e:
            print(f'[WARNING] Recall failed: {e}')
            # Fallback
            recall_text = f'\n\n{output_text}<memory>No record found.</memory>\n\n'
            prompt += recall_text
            cnt += 1
            print(recall_text)
    else:
        # No action, just continue
        prompt += output_text
        cnt += 1
        print(output_text)
