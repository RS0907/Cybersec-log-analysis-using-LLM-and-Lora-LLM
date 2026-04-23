import os
import json
import random
import time
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = Path(__file__).parent.parent  # Points to Lora LLM/
DATA_DIR = BASE_DIR / "datasets"
# train_gemma.py saves to ./lora_gemma_v3_final (run from Lora LLM/gemma2/)
LORA_DIR = BASE_DIR / "gemma2" / "lora_gemma_v3_final"
TEST_FILE = DATA_DIR / "test.jsonl"
OUTPUT_REPORT = BASE_DIR / "metrics" / "lora_gemma_report.json"
NUM_SAMPLES = 200

base_model_name = "unsloth/gemma-2-2b-it"
lora_dir = str(LORA_DIR)

if not os.path.exists(lora_dir):
    print(f"Error: {lora_dir} not found. Please train the model first.")
    exit(1)

print("1. Loading Tokenizer and Base Model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 
)

print("2. Fusing LoRA Weights...")
model = PeftModel.from_pretrained(base_model, lora_dir)
model.eval()

print("3. Loading Test Dataset...")
try:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"Error: {TEST_FILE} not found!")
    exit(1)

random.seed(42)
test_data = random.sample(all_data, min(NUM_SAMPLES, len(all_data)))

print(f"\n4. Starting Inference against Gemma LoRA...")
results = []
metrics = {
    "total": len(test_data),
    "valid_json": 0,
    "attack_type_correct": 0,
    "severity_correct": 0,
    "total_time": 0
}

start_time = time.time()

for i, row in enumerate(tqdm(test_data)):
    raw_input = row["input"]
    expected_output = row["output"]
    
    try:
        expected_json = json.loads(expected_output)
    except json.JSONDecodeError:
        continue

    messages = [
        {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{raw_input}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    is_valid_json = False
    attack_correct = False
    severity_correct = False
    predicted_json = {}
    
    try:
        clean_pred = prediction.replace("```json", "").replace("```", "").strip()
        start_idx = clean_pred.find("{")
        end_idx = clean_pred.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            clean_pred = clean_pred[start_idx:end_idx]
            predicted_json = json.loads(clean_pred)
            is_valid_json = True
            metrics["valid_json"] += 1
            
            p_atk = str(predicted_json.get("attack_type", "")).strip().lower()
            e_atk = str(expected_json.get("attack_type", "")).strip().lower()
            if e_atk and e_atk == p_atk:
                attack_correct = True
                metrics["attack_type_correct"] += 1
                
            p_sev = str(predicted_json.get("severity", "")).strip().lower()
            e_sev = str(expected_json.get("severity", "")).strip().lower()
            if e_sev and e_sev == p_sev:
                severity_correct = True
                metrics["severity_correct"] += 1
    except json.JSONDecodeError:
        pass
        
    results.append({
        "input": raw_input,
        "expected": expected_json,
        "predicted": predicted_json if is_valid_json else prediction,
        "is_valid_json": is_valid_json,
        "attack_correct": attack_correct,
        "severity_correct": severity_correct
    })

metrics["total_time"] = time.time() - start_time

print("\n" + "="*50)
print(f"       GEMMA LORA EVALUATION METRICS       ")
print("="*50)
print(f"Total Samples Evaluated : {len(results)}")
if len(results) > 0:
    print(f"Valid JSON Output Rate  : {(metrics['valid_json'] / len(results)) * 100:.2f}%")
    if metrics["valid_json"] > 0:
        print(f"Attack Type Accuracy    : {(metrics['attack_type_correct'] / metrics['valid_json']) * 100:.2f}%")
        print(f"Severity Accuracy       : {(metrics['severity_correct'] / metrics['valid_json']) * 100:.2f}%")
print(f"Total Time Taken        : {metrics['total_time']:.2f} seconds")
print("="*50)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    json.dump({"metrics": metrics, "detailed_results": results}, f, indent=4)
print(f"Granular results seamlessly exported to {OUTPUT_REPORT} for the Dashboard!")
