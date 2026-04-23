import os
import json
import random
import time
from tqdm import tqdm
from pathlib import Path
import sys
import ollama

# --- Configuration ---
NUM_SAMPLES = 200
TEST_FILE = Path(__file__).parent.parent / "Lora LLM" / "datasets" / "test.jsonl"
OUTPUT_DIR = Path(__file__).parent / "zero_shot_reports"

# Make sure this perfectly maps the exact Zero-Shot models to match the LoRA sizes!
MODELS_TO_EVALUATE = {
    "phi3:mini": "zero_shot_phi4_report.json",   # 3.8B match to prevent OOM
    "llama3.2": "zero_shot_llama3_report.json",    # 3B match
    "gemma2:2b": "zero_shot_gemma_report.json"    # 2B match
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"1. Loading test dataset ({TEST_FILE.name})...")
try:
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"Error: {TEST_FILE} not found!")
    exit(1)

# MUST match evaluate_metrics.py seed to compare the exact same 200 rows!
random.seed(42) 
if len(all_data) > NUM_SAMPLES:
    test_data = random.sample(all_data, NUM_SAMPLES)
else:
    test_data = all_data

print(f"Selected {len(test_data)} random samples for evaluation.")

# The system prompt we give the Zero-Shot model so it knows WHAT we are looking for.
SYSTEM_PROMPT = """
You are a cybersecurity JSON generator.
STRICT RULES:
- Output ONLY a valid JSON object.
- NO explanations, NO markdown block formatting.
- Keys must perfectly match exactly: "attack_type", "severity".
"""

for model_name, output_filename in MODELS_TO_EVALUATE.items():
    print(f"\n" + "="*50)
    print(f"2. Starting Inference Loop: {model_name}...")
    print("="*50)
    
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

        user_prompt = f"Analyze this network traffic flow and provide the threat classification JSON:\n{raw_input}"
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            prediction = response.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"\nOllama API error on row {i}: {e}. Ensure '{model_name}' is installed on Ollama.")
            break
        
        is_valid_json = False
        attack_correct = False
        severity_correct = False
        predicted_json = {}
        
        try:
            clean_pred = prediction.replace("```json", "").replace("```", "").strip()
            predicted_json = json.loads(clean_pred)
            is_valid_json = True
            metrics["valid_json"] += 1
            
            # Compare strings case-insensitively
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
            pass # Invalid JSON format generated
            
        results.append({
            "input": row["input"],
            "expected": expected_json,
            "predicted": predicted_json if is_valid_json else prediction,
            "is_valid_json": is_valid_json,
            "attack_correct": attack_correct,
            "severity_correct": severity_correct
        })

    metrics["total_time"] = time.time() - start_time
    output_path = OUTPUT_DIR / output_filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "detailed_results": results}, f, indent=4)
        
    if len(results) > 0:
        print(f"\n[Finished {model_name}] -> Valid JSON: {(metrics['valid_json'] / len(results)) * 100:.2f}% | Saved to {output_filename}")
    else:
        print(f"\n[Finished {model_name}] -> Failed or Skipped (Model likely not downloaded on Ollama).")
