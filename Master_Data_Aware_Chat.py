import pandas as pd
import ollama
import json
import random
import os
import sys
from pathlib import Path

# --- Configuration ---
CSV_PATH = Path("Lora LLM/datasets/unsw_nb15.csv")
TEST_JSONL = Path("Lora LLM/datasets/test.jsonl")

def get_dataset_summary():
    print("📊 Analyzing dataset for 'Data-Aware' memory...")
    if not CSV_PATH.exists():
        return "Dataset CSV not found."
    
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    attack_counts = df['attack_cat'].value_counts().to_dict()
    severity_counts = df['label'].value_counts().to_dict() # 0 = Normal, 1 = Attack
    
    # Calculate "Most Dangerous"
    # In UNSW-NB15, we can look at which attack categories have the most 'label=1'
    danger_attacks = df[df['label'] == 1]['attack_cat'].value_counts().head(3).index.tolist()
    
    summary = f"""
    DATASET CONTEXT (UNSW-NB15):
    - Total Network Flows Analyzed: {total_rows}
    - Attack Categories Found: {', '.join(attack_counts.keys())}
    - Most Frequent Attacks: {', '.join(list(attack_counts.keys())[:3])}
    - Most Dangerous/Frequent Threats: {', '.join(danger_attacks)}
    - Dataset Balance: {severity_counts.get(0, 0)} Normal flows vs {severity_counts.get(1, 0)} Attack flows.
    """
    return summary

def main():
    summary_context = get_dataset_summary()
    
    # Load test examples for 'r' feature
    all_examples = []
    if TEST_JSONL.exists():
        with open(TEST_JSONL, "r", encoding="utf-8") as f:
            all_examples = [json.loads(line) for line in f]

    print("\n" + "="*60)
    print(" 🛡️  MASTER DATA-AWARE CYBER-CHAT (ZERO-SHOT) 🛡️ ")
    print(" This model now 'knows' your dataset statistics!")
    print("="*60)

    # Pick a model
    models = ["llama3.2", "gemma2:2b", "phi3:mini"]
    print("\nChoose Base Model:")
    for i, m in enumerate(models):
        print(f" [{i+1}] {m}")
    
    choice = input("\nEnter choice (1-3): ").strip()
    model_name = models[int(choice)-1] if choice.isdigit() and int(choice) in [1,2,3] else "llama3.2"

    system_prompt = f"""
    {summary_context}
    
    You are a Cybersecurity Expert AI. You have been trained on the UNSW-NB15 dataset.
    STRICT RULES:
    1. If asked about the dataset, use the DATASET CONTEXT provided above.
    2. If asked to analyze a specific flow, output ONLY a JSON object with 'attack_type' and 'severity'.
    3. Be professional and concise.
    """

    print(f"\n--- Connected to {model_name} with Data-Aware Memory ---")
    print("Try asking: 'What is the most dangerous attack in our data?'")
    print("Or press 'r' for a random flow analysis.")

    while True:
        print("\n" + "-"*30)
        user_input = input("[You]: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            break
        
        if user_input.lower() == 'r':
            if not all_examples:
                print("No test examples found.")
                continue
            ex = random.choice(all_examples)
            user_input = f"Analyze this flow and provide JSON:\n{ex['input']}"
            print(f"\n[Random Flow Selected]:\n{ex['input']}")

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            print(f"\n[{model_name}]:\n{response['message']['content']}")
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
