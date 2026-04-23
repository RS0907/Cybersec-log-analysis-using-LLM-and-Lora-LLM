import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_FILE = BASE_DIR / "Lora" / "attack_accuracy_comparison.png"

EVALUATION_REPORTS = {
    # Base Models
    "Phi-4 (LLM)": BASE_DIR / "backend" / "outputs" / "zero_shot_phi4_report.json",
    "Llama-3 (LLM)": BASE_DIR / "backend" / "outputs" / "zero_shot_llama3_report.json",
    "Gemma-2 (LLM)": BASE_DIR / "backend" / "outputs" / "zero_shot_gemma_report.json",
    
    # LoRA Models
    "Phi-4 (LORA LLM)": BASE_DIR / "Lora" / "lora_evaluation_report.json",
    "Llama-3 (LORA LLM)": BASE_DIR / "Lora" / "lora_llama3_report.json",
    "Gemma (LORA LLM)": BASE_DIR / "Lora" / "lora_gemma_report.json"
}

def get_attack_acc(report_path):
    if not report_path.exists():
        return 0
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    total = max(metrics.get("valid_json", 1), 1)  
    return (metrics.get("attack_type_correct", 0) / total) * 100

models = list(EVALUATION_REPORTS.keys())
accuracies = [get_attack_acc(path) for path in EVALUATION_REPORTS.values()]

colors = ['#d9534f', '#d9534f', '#d9534f', '#5cb85c', '#5cb85c', '#5cb85c']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors)

plt.ylabel('Attack Type Accuracy (%)', fontsize=12)
plt.title('Zero-Shot LLM vs. Fine-Tuned LORA LLM (Cyber Threat Detection)', fontsize=14, pad=15)
plt.xticks(rotation=25, ha='right', fontsize=10)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add exact percentage labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Isolated Accuracy Graph saved to: {OUTPUT_FILE}")
