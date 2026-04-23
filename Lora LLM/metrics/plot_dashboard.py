"""
Cap-Code — LLM Performance Dashboard
Reads the 3 real LoRA JSON evaluation reports and generates a
publication-quality 4-panel comparison chart.

Run from Cap-Code root:
    python "Lora LLM/metrics/plot_dashboard.py"
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ─────────────────────────────────────────────────────────
# 1.  LOAD REAL EVALUATION REPORTS
# ─────────────────────────────────────────────────────────
BASE = Path(__file__).parent   # Lora LLM/metrics/

reports = {
    "Phi-4 Mini\n(LoRA)":    BASE / "lora_evaluation_report.json",
    "Llama-3.2 3B\n(LoRA)":  BASE / "lora_llama3_report.json",
    "Gemma-2 2B\n(LoRA)":    BASE / "lora_gemma_report.json",
}

models, valid_json_pct, attack_acc, severity_acc = [], [], [], []

for label, path in reports.items():
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    m   = data["metrics"]
    tot = m["total"]
    vj  = m["valid_json"]
    ac  = m["attack_type_correct"]
    sc  = m["severity_correct"]
    models.append(label)
    valid_json_pct.append(round(vj / tot * 100, 1))
    attack_acc.append(round(ac / vj  * 100, 1) if vj > 0 else 0.0)
    severity_acc.append(round(sc / vj * 100, 1) if vj > 0 else 0.0)

# Zero-Shot baseline numbers (Llama-3 8B, 200 flows, no fine-tuning)
# Based on the project's documented zero-shot run.
ZS_ATTACK   = 18.5
ZS_SEVERITY = 31.0
ZS_JSON     = 72.0

# ─────────────────────────────────────────────────────────
# 2.  STYLE
# ─────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
TEXT_MAIN = "#e6edf3"
TEXT_SUB  = "#8b949e"

PHI_C     = "#7c3aed"
LLAMA_C   = "#0ea5e9"
GEMMA_C   = "#10b981"
ZS_C      = "#f59e0b"
MODEL_COLORS = [PHI_C, LLAMA_C, GEMMA_C]

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "text.color":       TEXT_MAIN,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_MAIN,
    "xtick.color":      TEXT_SUB,
    "ytick.color":      TEXT_SUB,
    "figure.facecolor": DARK_BG,
    "grid.color":       BORDER,
    "grid.linestyle":   "--",
    "grid.alpha":       0.55,
})

# ─────────────────────────────────────────────────────────
# 3.  FIGURE  (2 × 2 grid)
# ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(DARK_BG)
gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30,
               left=0.07, right=0.96, top=0.88, bottom=0.10)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
# ax4 will be polar – added below

x  = np.arange(len(models))
BW = 0.32

# ── helpers ──────────────────────────────────────────────
def style_ax(ax, title, ylabel, ylim=(0, 115)):
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=TEXT_MAIN, pad=10)
    ax.set_ylabel(ylabel, color=TEXT_SUB, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def add_bar_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.9,
                f"{h}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=TEXT_MAIN)

def add_zs_line(ax, val, label="Zero-Shot baseline"):
    ax.axhline(val, color=ZS_C, linewidth=1.6, linestyle="--", alpha=0.85, zorder=0)
    ax.text(len(models) - 0.52, val + 1.2, f"ZS  {val}%",
            color=ZS_C, fontsize=8, ha="right", va="bottom")

# ─────────────────────────────────────────────────────────
# PANEL 1  —  Attack Type Accuracy
# ─────────────────────────────────────────────────────────
bars1 = ax1.bar(x, attack_acc, BW * 1.9, color=MODEL_COLORS,
                edgecolor=DARK_BG, linewidth=0.8, zorder=3)
add_bar_labels(ax1, bars1)
add_zs_line(ax1, ZS_ATTACK)
style_ax(ax1, "[1]  Attack Type Accuracy  (LoRA vs Zero-Shot)", "Accuracy (%)")

# ─────────────────────────────────────────────────────────
# PANEL 2  —  Severity Accuracy
# ─────────────────────────────────────────────────────────
bars2 = ax2.bar(x, severity_acc, BW * 1.9, color=MODEL_COLORS,
                edgecolor=DARK_BG, linewidth=0.8, zorder=3)
add_bar_labels(ax2, bars2)
add_zs_line(ax2, ZS_SEVERITY)
style_ax(ax2, "[2]  Severity Classification Accuracy  (LoRA vs Zero-Shot)", "Accuracy (%)")

# ─────────────────────────────────────────────────────────
# PANEL 3  —  Valid JSON Rate  (grouped: LoRA vs ZS)
# ─────────────────────────────────────────────────────────
lora_bars = ax3.bar(x - BW / 2, valid_json_pct, BW,
                    color=MODEL_COLORS, edgecolor=DARK_BG,
                    linewidth=0.8, zorder=3, label="LoRA Fine-Tuned")
zs_bars   = ax3.bar(x + BW / 2, [ZS_JSON] * 3, BW,
                    color=ZS_C, edgecolor=DARK_BG,
                    linewidth=0.8, alpha=0.75, zorder=3, label="Zero-Shot (Llama-3 8B)")
add_bar_labels(ax3, lora_bars)
add_bar_labels(ax3, zs_bars)
style_ax(ax3, "[3]  Valid JSON Output Rate  (LoRA vs Zero-Shot)", "Rate (%)")
ax3.legend(loc="lower right", fontsize=8,
           facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_MAIN)

# ─────────────────────────────────────────────────────────
# PANEL 4  —  Radar / Spider chart
# ─────────────────────────────────────────────────────────
categories = ["Attack\nAccuracy", "Severity\nAccuracy", "Valid JSON\nRate"]
N      = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax4 = fig.add_subplot(gs[1, 1], polar=True)
ax4.set_facecolor(PANEL_BG)
ax4.spines["polar"].set_color(BORDER)
ax4.tick_params(colors=TEXT_SUB, labelsize=8)

model_labels_flat = ["Phi-4 Mini (LoRA)", "Llama-3.2 3B (LoRA)", "Gemma-2 2B (LoRA)"]
for i, (label, color) in enumerate(zip(model_labels_flat, MODEL_COLORS)):
    vals  = [attack_acc[i], severity_acc[i], valid_json_pct[i]]
    vals += vals[:1]
    ax4.plot(angles, vals, color=color, linewidth=2.2, label=label)
    ax4.fill(angles, vals, color=color, alpha=0.11)

# Zero-shot polygon
zs_vals  = [ZS_ATTACK, ZS_SEVERITY, ZS_JSON]
zs_vals += zs_vals[:1]
ax4.plot(angles, zs_vals, color=ZS_C, linewidth=1.8,
         linestyle="--", label="Zero-Shot (Llama-3 8B)")
ax4.fill(angles, zs_vals, color=ZS_C, alpha=0.06)

ax4.set_thetagrids(np.degrees(angles[:-1]), categories,
                   color=TEXT_MAIN, fontsize=9)
ax4.set_ylim(0, 100)
ax4.yaxis.set_tick_params(labelcolor=TEXT_SUB, labelsize=7)
ax4.set_title("[4]  Overall Model Radar", fontsize=11,
              fontweight="bold", color=TEXT_MAIN, pad=18)
ax4.legend(loc="upper right", bbox_to_anchor=(1.42, 1.14),
           fontsize=7.5, facecolor=PANEL_BG,
           edgecolor=BORDER, labelcolor=TEXT_MAIN)

# ─────────────────────────────────────────────────────────
# 4.  MASTER TITLE + BOTTOM LEGEND
# ─────────────────────────────────────────────────────────
fig.text(0.50, 0.955,
         "Cap-Code  ·  QLoRA Fine-Tuned LLMs vs Zero-Shot — Network Threat Detection",
         ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT_MAIN)
fig.text(0.50, 0.925,
         "UNSW-NB15 Dataset  ·  n = 200 test flows  ·  seed = 42  ·  "
         "Phi-4 Mini | Llama-3.2 3B | Gemma-2 2B",
         ha="center", va="center", fontsize=9, color=TEXT_SUB)

legend_handles = [
    mpatches.Patch(color=PHI_C,   label="Phi-4 Mini Instruct (LoRA)"),
    mpatches.Patch(color=LLAMA_C, label="Llama-3.2 3B Instruct (LoRA)"),
    mpatches.Patch(color=GEMMA_C, label="Gemma-2 2B IT (LoRA)"),
    mpatches.Patch(color=ZS_C, alpha=0.75, label="Zero-Shot Baseline — Llama-3 8B"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9, facecolor=PANEL_BG, edgecolor=BORDER,
           labelcolor=TEXT_MAIN, bbox_to_anchor=(0.5, 0.005))

# ─────────────────────────────────────────────────────────
# 5.  SAVE
# ─────────────────────────────────────────────────────────
out = BASE / "cap_code_dashboard.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
print(f"Dashboard saved -> {out}")
