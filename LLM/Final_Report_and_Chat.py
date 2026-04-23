import pandas as pd
import json
import time
import random
import ollama
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "Lora LLM" / "datasets" / "unsw_nb15.csv"
OUTPUT_REPORT = BASE_DIR / "LLM" / "zero_shot_reports" / "full_dataset_report.json"
NUM_SAMPLES = 1000 # Increased to 1000 for a deeper 'Processing' phase

def process_entire_dataset(model_name):
    print(f"\n📊 PHASE 1: PROCESSING {NUM_SAMPLES} SAMPLES FROM DATASET USING {model_name}...")
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found!")
        return None

    df = pd.read_csv(CSV_PATH)
    df_sample = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)
    
    events = []
    print(f"Analyzing trends via {model_name}...")
    
    for idx, (_, row) in enumerate(tqdm(df_sample.iterrows(), total=len(df_sample))):
        # Concise context for processing speed
        flow_context = f"Proto: {row.get('proto')}, Srv: {row.get('service')}, Rate: {row.get('rate')}, SBytes: {row.get('sbytes')}"
        
        # We ask for a quick category to build the report
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": f"Classify this flow as Normal or Attack type: {flow_context}"}]
        )
        prediction = response['message']['content'].strip()
        events.append({"id": idx, "prediction": prediction, "actual": row.get('attack_cat')})

    # Build Global Report for the Chat phase
    report = {
        "total_analyzed": NUM_SAMPLES,
        "global_dataset_stats": {
            "total_rows": len(df),
            "attack_types_in_csv": df['attack_cat'].unique().tolist(),
            "top_protocol": df['proto'].value_counts().idxmax()
        },
        "sample_findings": events[:10] # Give the LLM some raw examples
    }
    
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    
    return report

def interactive_qa(report, model_name):
    print("\n" + "="*60)
    print(f" 🤖 DATASET-AWARE ANALYST CHAT ({model_name}) 🤖 ")
    print(f" The LLM has now 'processed' {NUM_SAMPLES} real-world samples.")
    print("="*60)
    
    qa_context = f"""
    You are a Senior Security Analyst. You just finished a processing run on the UNSW-NB15 dataset.
    SUMMARY OF YOUR FINDINGS:
    - Total CSV Rows: {report['global_dataset_stats']['total_rows']}
    - Unique Attacks Seen: {report['global_dataset_stats']['attack_types_in_csv']}
    - Dominant Protocol: {report['global_dataset_stats']['top_protocol']}
    
    Answer the user's questions about the dataset trends or specific threats.
    """

    while True:
        print("\n" + "-"*30)
        user_input = input("[You]: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": qa_context},
                    {"role": "user", "content": user_input}
                ]
            )
            print(f"\n[{model_name}]:\n{response['message']['content']}")
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    # Pick a model
    models = ["llama3.2", "gemma2:2b", "phi3:mini"]
    print("\n" + "="*50)
    print(" Choose Base Model for Processing & Chat:")
    for i, m in enumerate(models):
        print(f" [{i+1}] {m}")
    
    choice = input("\nEnter choice (1-3): ").strip()
    model_name = models[int(choice)-1] if choice.isdigit() and int(choice) in [1,2,3] else "llama3.2"

    full_report = process_entire_dataset(model_name)
    if full_report:
        interactive_qa(full_report, model_name)
