import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_FILE = BASE_DIR / "Lora" / "training_metrics_analysis.png"

TRAINING_RUNS = {
    "Phi-4 (LORA LLM)": BASE_DIR / "Lora" / "lora_unsw_v2" / "checkpoint-1000" / "trainer_state.json",
    "Llama-3 (LORA LLM)": BASE_DIR / "llama-lora" / "lora_llama_outputs" / "checkpoint-1000" / "trainer_state.json", 
    "Gemma (LORA LLM)": BASE_DIR / "gemma-lora" / "lora_gemma_outputs" / "checkpoint-1000" / "trainer_state.json"
}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
colors = ["royalblue", "darkorange", "forestgreen"]

# ==========================================
# Plot 1: Mean Token Accuracy
# ==========================================
for idx, (model_name, state_path) in enumerate(TRAINING_RUNS.items()):
    if not state_path.exists(): continue
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        steps, acc = [], []
        for log in data.get("log_history", []):
            if "step" in log and "mean_token_accuracy" in log:
                steps.append(log["step"])
                # Convert to percentage for readability
                acc.append(log["mean_token_accuracy"] * 100)
        if steps:
            ax1.plot(steps, acc, label=model_name, color=colors[idx], linewidth=2.5)

ax1.set_title("Training Token Accuracy Timeline", fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel("Training Steps", fontsize=12)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="lower right")

# ==========================================
# Plot 2: Learning Rate Schedule
# ==========================================
for idx, (model_name, state_path) in enumerate(TRAINING_RUNS.items()):
    if not state_path.exists(): continue
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        steps, lr = [], []
        for log in data.get("log_history", []):
            if "step" in log and "learning_rate" in log:
                steps.append(log["step"])
                lr.append(log["learning_rate"])
        if steps:
            ax2.plot(steps, lr, label=model_name, color=colors[idx], linewidth=2.5)

ax2.set_title("Learning Rate Decay (Cosine Schedule)", fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel("Training Steps", fontsize=12)
ax2.set_ylabel("Learning Rate", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper right")

# ==========================================
# Plot 3: Gradient Norm Stability
# ==========================================
for idx, (model_name, state_path) in enumerate(TRAINING_RUNS.items()):
    if not state_path.exists(): continue
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        steps, grad = [], []
        for log in data.get("log_history", []):
            if "step" in log and "grad_norm" in log:
                steps.append(log["step"])
                grad.append(log["grad_norm"])
        if steps:
            ax3.plot(steps, grad, label=model_name, color=colors[idx], alpha=0.7, linewidth=1.5)

ax3.set_title("Neural Network Stability (Gradient Norm)", fontsize=14, fontweight='bold', pad=10)
ax3.set_xlabel("Training Steps", fontsize=12)
ax3.set_ylabel("Gradient Norm", fontsize=12)
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"\n🎉 3 Academic Line Graphs generated successfully!")
print(f"Saved to: {OUTPUT_FILE}")
