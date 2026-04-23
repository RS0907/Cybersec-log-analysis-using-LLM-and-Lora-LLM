import ollama
import sys
import json
import random
from pathlib import Path

def interactive_chat():
    # Load dataset for easy testing
    BASE_DIR = Path(__file__).parent.parent
    DATASET_PATH = BASE_DIR / "Lora LLM" / "datasets" / "test.jsonl"
    all_examples = []
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            all_examples = [json.loads(line) for line in f]

    print("\n" + "="*50)
    print(" 🌍 MASTER BASE LLM CHAT (ZERO-SHOT) 🌍")
    print("="*50)
    
    # List models available in Ollama
    try:
        models = ["llama3.2", "gemma2:2b", "phi3:mini"]
        print("\nSelect a Base Model to test:")
        for i, m in enumerate(models):
            print(f" [{i+1}] {m}")
            
        choice = input("\nEnter number (1-3): ").strip()
        if not choice.isdigit() or int(choice) not in [1, 2, 3]:
            print("Invalid choice. Defaulting to Llama3.2")
            model_name = "llama3.2"
        else:
            model_name = models[int(choice)-1]
            
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return

    print(f"\n--- Chatting with BASE {model_name} ---")
    print("Type 'exit' to switch models or quit.")
    
    system_prompt = "You are a cybersecurity JSON generator. Output ONLY valid JSON with keys 'attack_type' and 'severity'."

    while True:
        print("\n" + "-"*30)
        print(" [Action]: Paste a flow, type 'r' for random dataset example, or 'exit'.")
        user_input = input("[Input]: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if user_input.lower() == 'r':
            if not all_examples:
                print("Error: Dataset test.jsonl not found!")
                continue
            example = random.choice(all_examples)
            user_input = example["input"]
            print(f"\n[Random Example Selected]:\n{user_input}")
            print(f"[Ground Truth]: {example['output']}")

        if not user_input:
            continue

        try:
            print(f"Asking {model_name}...")
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{user_input}"}
                ]
            )
            print(f"\n[{model_name} Prediction]:\n{response['message']['content']}")
        except Exception as e:
            print(f"Error during inference: {e}")
            print(f"Tip: Make sure you have run 'ollama pull {model_name}' first!")
            break

if __name__ == "__main__":
    interactive_chat()
