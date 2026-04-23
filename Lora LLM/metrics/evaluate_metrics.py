import os
import json
import random
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
NUM_SAMPLES = 200  # Number of rows to test for the initial run to save time!
BASE_MODEL = "microsoft/phi-4-mini-instruct"

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent  # Points to Cap-Code/
LORA_DIR = str(BASE_DIR / "phi4" / "lora_unsw_v3_final")
TEST_FILE = str(BASE_DIR / "datasets" / "test.jsonl")
OUTPUT_REPORT = str(BASE_DIR / "metrics" / "lora_evaluation_report.json")

print(f"1. Loading test dataset ({TEST_FILE})...")
try:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"Error: {TEST_FILE} not found! Ensure you are in the Lora/ directory.")
    exit(1)

# Randomly sample the dataset to save inference time on local GPU
random.seed(42) # Fixed seed for reproducibility so you test the same 200 rows every time
if len(all_data) > NUM_SAMPLES:
    test_data = random.sample(all_data, NUM_SAMPLES)
else:
    test_data = all_data

print(f"Selected {len(test_data)} random samples for evaluation.")

print("\n2. Loading tokenizer and models (4-bit)...")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Must load in 4-bit because adapter was trained over a 4-bit base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

print(f"Loading LoRA adapter from {LORA_DIR}...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

print("\n3. Starting Inference Loop...")
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
    
    # Parse the ground truth JSON
    try:
        expected_json = json.loads(expected_output)
    except json.JSONDecodeError:
        print(f"Warning: Ground truth row {i} has invalid JSON. Skipping.")
        continue

    # Format the prompt accurately
    messages = [
        {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{raw_input}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1, # Keep it deterministic for evaluation
            do_sample=False
        )
        
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # ---------------------------
    # Evaluate JSON format and Accuracy
    # ---------------------------
    is_valid_json = False
    attack_correct = False
    severity_correct = False
    predicted_json = {}
    
    try:
        # Sometimes models wrap outputs in code blocks
        clean_pred = prediction.replace("```json", "").replace("```", "").strip()
        predicted_json = json.loads(clean_pred)
        is_valid_json = True
        metrics["valid_json"] += 1
        
        # Grading Check
        if expected_json.get("attack_type", "").lower() == predicted_json.get("attack_type", "").lower():
            attack_correct = True
            metrics["attack_type_correct"] += 1
            
        if expected_json.get("severity", "").lower() == predicted_json.get("severity", "").lower():
            severity_correct = True
            metrics["severity_correct"] += 1
            
    except json.JSONDecodeError:
        pass # Invalid JSON format generated
        
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
print("             FINAL EVALUATION METRICS             ")
print("="*50)
print(f"Total Samples Evaluated : {metrics['total']}")
if metrics["total"] > 0:
    print(f"Valid JSON Output Rate  : {(metrics['valid_json'] / metrics['total']) * 100:.2f}%")
    
    if metrics["valid_json"] > 0:
        print(f"Attack Type Accuracy    : {(metrics['attack_type_correct'] / metrics['valid_json']) * 100:.2f}%")
        print(f"Severity Accuracy       : {(metrics['severity_correct'] / metrics['valid_json']) * 100:.2f}%")
print(f"Total Time Taken        : {metrics['total_time']:.2f} seconds")
print("="*50)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    json.dump({"metrics": metrics, "detailed_results": results}, f, indent=4)
print(f"Detailed granular results saved to {OUTPUT_REPORT}")
