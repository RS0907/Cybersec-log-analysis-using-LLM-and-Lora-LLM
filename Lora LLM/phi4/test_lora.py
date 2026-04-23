import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import random
from pathlib import Path

base_model_name = "microsoft/phi-4-mini-instruct"
# 1. We point to the actual model we just finished training (lora_unsw_final)
lora_dir = "./lora_unsw_v3_final" 

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading base model in 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 # Safely handle bfloat16 parameters
)

print(f"Loading LoRA adapter from {lora_dir}...")
model = PeftModel.from_pretrained(base_model, lora_dir)
model.eval()

# Load dataset for easy testing
BASE_DIR = Path(__file__).parent.parent.parent # Cap-Code/
DATASET_PATH = BASE_DIR / "Lora LLM" / "datasets" / "test.jsonl"
all_examples = []
if DATASET_PATH.exists():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_examples = [json.loads(line) for line in f]

print("\n" + "="*50)
print(" 🚀 INTERACTIVE PHI-4 LORA CHAT LAUNCHED 🚀")
print(" Type 'r' for random example, or 'exit' to quit.")
print("="*50)

while True:
    print("\n" + "-"*30)
    print(" [Action]: Paste a flow, type 'r' for random dataset example, or 'exit'.")
    raw_network_flow = input("[Input]: ").strip()
    
    if raw_network_flow.lower() in ['exit', 'quit']:
        print("Ending interactive session. Great job on the training!")
        break
        
    if raw_network_flow.lower() == 'r':
        if not all_examples:
            print("Error: Dataset test.jsonl not found!")
            continue
        example = random.choice(all_examples)
        raw_network_flow = example["input"]
        print(f"\n[Random Example Selected]:\n{raw_network_flow}")
        print(f"[Ground Truth]: {example['output']}")

    if not raw_network_flow:
        continue

    # We MUST format this exactly how the model was trained
    messages = [
        {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{raw_network_flow}"}
    ]

    # This renders the chat template strictly required by modern Phi-4 Instruct models
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1
        )

    # Slice off the prompt block to extract only what the model generated inside the Assistant's turn
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\n[LoRA Prediction]:\n{prediction}")