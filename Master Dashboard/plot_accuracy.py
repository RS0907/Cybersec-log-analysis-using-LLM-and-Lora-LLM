import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths to the evaluation reports
BASE_DIR = Path(__file__).parent.parent
LORA_REPORT = BASE_DIR / "Lora" / "lora_evaluation_report.json"
ZERO_SHOT_REPORT = BASE_DIR / "backend" / "outputs" / "zero_shot_evaluation_report.json"

def calculate_rates(report_path):
    if not report_path.exists():
        return None
        
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    total = metrics.get("valid_json", 1)  # avoid div by zero, measure against valid json
    if total == 0:
        total = 1
        
    attack_acc = (metrics.get("attack_type_correct", 0) / total) * 100
    severity_acc = (metrics.get("severity_correct", 0) / total) * 100
    valid_format = (metrics.get("valid_json", 0) / metrics.get("total", 1)) * 100
    
    return [attack_acc, severity_acc, valid_format]

print("Scanning for evaluation reports...")

models = []
attack_accs = []
severity_accs = []
format_rates = []

# Fetch LoRA 
lora_metrics = calculate_rates(LORA_REPORT)
if lora_metrics:
    models.append("Phi-4-Mini (LoRA)")
    attack_accs.append(lora_metrics[0])
    severity_accs.append(lora_metrics[1])
    format_rates.append(lora_metrics[2])
    print(f"✅ Found LoRA Report: {LORA_REPORT.name}")
else:
    print(f"❌ Missing LoRA Report: {LORA_REPORT}")

# Fetch Zero-Shot
zero_metrics = calculate_rates(ZERO_SHOT_REPORT)
if zero_metrics:
    models.append("Llama3 (Zero-Shot)")
    attack_accs.append(zero_metrics[0])
    severity_accs.append(zero_metrics[1])
    format_rates.append(zero_metrics[2])
    print(f"✅ Found Zero-Shot Report: {ZERO_SHOT_REPORT.name}")
else:
    print(f"⚠️ Missing Zero-Shot Report: {ZERO_SHOT_REPORT}")
    print("   (Run evaluate_zero_shot.py to generate this for a full comparison)")

if not models:
    print("No reports found to plot! Please run evaluate_metrics.py or evaluate_zero_shot.py first.")
    exit()

# ----------- PLOTTING -----------
# Set up the bar chart
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, attack_accs, width, label='Attack Type Accuracy', color='#4C72B0')
rects2 = ax.bar(x, severity_accs, width, label='Severity Accuracy', color='#55A868')
rects3 = ax.bar(x + width, format_rates, width, label='Valid JSON Format Rate', color='#C44E52')

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Model Performance Evaluation on UNSW-NB15', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 115) 
ax.legend(loc='upper right')

# Add text labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
OUTPUT_FILE = BASE_DIR / "Lora" / "accuracy_comparison.png"
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"\n🎉 Graph generated successfully!")
print(f"Saved to: {OUTPUT_FILE}")
