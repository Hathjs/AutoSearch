# ---

**AutoSearch: System Design Specification for AI Assistant**

## **1\. Project Overview & Objective**

We are upgrading an existing RL-based Search Agent (based on Search-R1 architecture) to AutoSearch V2.  
The core innovation is the "Decoupled Dual-Channel Architecture":

1. **Channel 1 (Internal):** Parametric Memory Retrieval (\<recall\> \-\> \<memory\>). Low cost, high speed, checks internal knowledge boundary.  
2. **Channel 2 (External):** Tool Use (\<search\> \-\> \<observation\>). High cost, high latency, used for dynamic info.

**Goal:** Modify the inference logic, prompt templates, and reward functions to support this dual-channel mechanism.

## ---

**2\. The Interaction Protocol (XML Definition)**

We enforce a strict XML-based action space. The model must strictly follow these tags.

| Component | Tag Pair | Origin | Description | Cost Model (RL) |
| :---- | :---- | :---- | :---- | :---- |
| **Reasoning** | \<think\>...\</think\> | Model | Chain-of-thought planning & reflection. | 0 (Base cost) |
| **Action A** | \<recall\>query\</recall\> | Model | **Internal Knowledge Retrieval.** Used for facts/definitions/sanity checks. | **Cost \= 0.0 (or very low)** |
| **Feedback A** | \<memory\>content\</memory\> | System | The result of recall. (In SFT: Generated or Hardcoded; In RL: Placeholder or Self-generated). | N/A |
| **Action B** | \<search\>query\</search\> | Model | **External API Call.** Used for real-time news/unknowns. | **Cost \= $$$ (Penalty)** |
| **Feedback B** | \<observation\>content\</observation\> | System | Result from Google/Bing API. | N/A |
| **Output** | \<answer\>content\</answer\> | Model | Final response to user. | N/A |

## ---

**3\. Engineering Implementation Details**

### **A. System Prompt Template (To replace existing prompt)**

The system prompt must explicitly define the two channels and the "First-Internal-Then-External" bias.

Python

SYSTEM\_PROMPT \= """You are AutoSearch, an intelligent agent with dual information retrieval capabilities.  
Please answer the user's question step-by-step.

\*\*Core Protocol & Action Space:\*\*  
1\. \*\*Reasoning (\`\<think\>\`):\*\* \- ALWAYS start with \`\<think\>\` to analyze the question.  
   \- Assess if you have sufficient internal knowledge.

2\. \*\*Internal Recall (\`\<recall\>\`):\*\* \- \*\*PRIORITY:\*\* If the question asks for static facts, definitions, or common sense, verify your memory first.  
   \- \*Syntax:\* \`\<recall\>keywords\</recall\>\`  
   \- \*System Feedback:\* Returns \`\<memory\>...\</memory\>\`

3\. \*\*External Search (\`\<search\>\`):\*\* \- \*\*SECONDARY:\*\* Use ONLY if internal memory is insufficient, uncertain (\`No record\`), or if the question requires real-time/private info.  
   \- \*Syntax:\* \`\<search\>keywords\</search\>\`  
   \- \*System Feedback:\* Returns \`\<observation\>...\</observation\>\`

4\. \*\*Termination:\*\*  
   \- Output final answer in \`\<answer\>...\</answer\>\`.

\*\*Trap/Hallucination Handling:\*\*  
\- If \`\<recall\>\` returns \`\<memory\>No record\</memory\>\` or contradicts the user's premise, you MUST refuse to answer or correct the premise. Do NOT fabricate information.

\*\*Example:\*\*  
User: "Who is the CEO of SpaceX?"  
Assistant: \<think\>I recall Elon Musk founded SpaceX.\</think\>\<recall\>SpaceX CEO\</recall\>\<memory\>Elon Musk is the CEO.\</memory\>\<answer\>Elon Musk.\</answer\>  
"""

### **B. Inference Logic (Python Regex Parsing)**

We need to modify the execution loop (e.g., inside generation.py or env.step) to handle the new \<recall\> tag.

**Key Logic:**

1. Parse for \<recall\> tags.  
2. If \<recall\> is found, **DO NOT** call Google API. Instead, return a "Virtual Memory" signal (or empty string/placeholder depending on setup).  
3. If \<search\> is found, call Google API.

Python

import re

def parse\_and\_execute\_step(model\_output):  
    """  
    Parses model output for actions.  
    Priority: Search \> Recall (if both exist, usually search takes precedence or execute sequentially).  
    """  
      
    \# 1\. Check for External Search  
    search\_match \= re.search(r"\<search\>(.\*?)\</search\>", model\_output, re.DOTALL)  
    if search\_match:  
        query \= search\_match.group(1).strip()  
        \# Call Google Search API  
        search\_results \= google\_search\_tool(query)   
        return f"\<observation\>{search\_results}\</observation\>"

    \# 2\. Check for Internal Recall (New Logic)  
    recall\_match \= re.search(r"\<recall\>(.\*?)\</recall\>", model\_output, re.DOTALL)  
    if recall\_match:  
        query \= recall\_match.group(1).strip()  
          
        \# NOTE: In a real system, this might query a VectorDB.  
        \# For AutoSearch "Parametric Memory" simulation:  
        \# We return a specific token indicating memory access was attempted.  
        \# Or in simple cases, allow the model to continue generating (Self-Correction).  
        \# Here we return a standardized system message for consistency.  
          
        return "\<memory\>Memory Access: Retrieval Complete.\</memory\>" 

    return None \# No action, assume End of Turn or Answer

### **C. Reward Function Design (RL Stage)**

Modify the compute\_score function to implement the **Pricing Mechanism** and **Consistency Check**.

Python

def compute\_reward(trajectory, ground\_truth, type\="normal"):  
    """  
    trajectory: Full string of the conversation.  
    ground\_truth: Target answer.  
    type: "normal" or "trap" (from dataset metadata).  
    """  
    reward \= 0.0  
      
    \# \--- 1\. Base Outcome Reward \---  
    prediction \= extract\_answer(trajectory)  
    if is\_correct(prediction, ground\_truth):  
        reward \+= 1.0  
    else:  
        reward \-= 1.0

    \# \--- 2\. Cost Penalty (The Economic Model) \---  
    \# We penalize \<search\> but NOT \<recall\>.  
    num\_searches \= len(re.findall(r"\<search\>", trajectory))  
    num\_recalls \= len(re.findall(r"\<recall\>", trajectory))  
      
    \# Search is expensive (e.g., $0.1)  
    reward \-= (num\_searches \* 0.1)  
      
    \# Recall is cheap/free (e.g., $0.0 or $0.01)  
    reward \-= (num\_recalls \* 0.0) 

    \# \--- 3\. Consistency/Safety Check (Crucial for Trap Questions) \---  
    \# If the question is a TRAP and model successfully Refused  
    if type \== "trap":  
        if is\_refusal(prediction):   
            reward \+= 1.0 \# Bonus for safety  
          
        \# Verify Process: Did it check memory first?  
        if "\<recall\>" in trajectory and "No record" in trajectory:  
            reward \+= 0.5 \# Process Reward for correct checking behavior

    return reward

## ---

**4\. SFT Data Construction Guide (JSONL Format)**

We need to construct train.jsonl with mixed data types to teach the model the dual-channel behavior.

### **Type A: Success Recall (Standard Knowledge)**

JSON

{  
  "messages": \[  
    {"role": "system", "content": "...(System Prompt)..."},  
    {"role": "user", "content": "What is the capital of France?"},  
    {"role": "assistant", "content": "\<think\>This is a basic geography fact. I should recall it.\</think\>\<recall\>capital of France\</recall\>\<memory\>Paris\</memory\>\<answer\>Paris is the capital of France.\</answer\>"}  
  \]  
}

### **Type B: External Search (Dynamic/Unknown)**

JSON

{  
  "messages": \[  
    {"role": "system", "content": "..."},  
    {"role": "user", "content": "What is the stock price of Apple right now?"},  
    {"role": "assistant", "content": "\<think\>Stock prices are real-time dynamic data. Internal memory is not sufficient.\</think\>\<search\>AAPL stock price today\</search\>\<observation\>AAPL is trading at $220.50...\</observation\>\<answer\>As of today, Apple stock is $220.50.\</answer\>"}  
  \]  
}

### **Type C: Trap Handling (The "Pivot" Logic)**

**Critical:** This teaches the model to verify premises.

JSON

{  
  "messages": \[  
    {"role": "system", "content": "..."},  
    {"role": "user", "content": "Who won the 2024 Turing Award?"},  
    {"role": "assistant", "content": "\<think\>I need to check if the 2024 Turing Award has been announced. Let me recall recent awards.\</think\>\<recall\>2024 Turing Award winner\</recall\>\<memory\>No record found.\</memory\>\<think\>My memory returns no record. This implies it might not be announced yet. I should not hallucinate a name.\</think\>\<answer\>The 2024 Turing Award winner has not been announced yet.\</answer\>"}  
  \]  
}

## ---

**5\. Execution Roadmap for AI Assistant**

1. **Refactor Regex:** Update the parser in the inference engine to support \<recall\>.  
2. **Generate Data:** Create the SFT dataset (10k samples) mixing the 3 types above.  
3. **Cold Start:** Run SFT for 1 epoch on Qwen-2.5-3B to teach format compliance.  
4. **RL Training:** Run PPO/GRPO using the modified Reward Function (with penalties).

---

**End of Specification**