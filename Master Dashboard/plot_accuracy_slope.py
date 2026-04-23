import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_FILE = BASE_DIR / "Lora" / "accuracy_slope_graph.png"

# The strictly extracted attack type precision metrics
models = ["Llama-3", "Gemma-2", "Phi-4"]
base_acc = [0.00, 0.50, 0.00]
lora_acc = [60.50, 62.50, 67.50]

fig, ax = plt.subplots(figsize=(8, 6))

colors = ['darkorange', 'forestgreen', 'royalblue']
markers = ['o', 's', '^']

# Draw the line graphs
for i in range(3):
    # This draws a line traveling from the Zero-Shot configuration to the LoRA configuration
    ax.plot(['Zero-Shot (Out-of-the-box)', 'LORA LLM (Fine-Tuned)'], [base_acc[i], lora_acc[i]], 
            marker=markers[i], markersize=10, linewidth=3.5, 
            color=colors[i], label=models[i])
    
    # Label the starting base value
    ax.annotate(f"{base_acc[i]:.1f}%", 
                xy=(0, base_acc[i]), xytext=(-15, 0),
                textcoords="offset points", ha='right', va='center',
                fontsize=11, fontweight='bold', color=colors[i])

    # Label the ending LoRA value
    ax.annotate(f"{lora_acc[i]:.1f}%", 
                xy=(1, lora_acc[i]), xytext=(15, 0),
                textcoords="offset points", ha='left', va='center',
                fontsize=11, fontweight='bold', color=colors[i])

ax.set_ylabel('Attack Type Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Intelligence Growth Trajectory (LLM -> LORA)', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(-5, 90)
ax.grid(True, axis='y', linestyle="--", alpha=0.5)

# Place legend cleanly
ax.legend(title="Architecture Matrix", fontsize=11, title_fontsize=12, loc='upper left')

# Expand margins to ensure text labels fit perfectly
plt.subplots_adjust(left=0.15, right=0.85)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Trajectory Line Graph saved to: {OUTPUT_FILE}")
