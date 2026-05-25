# 🛡️ Cybersec Log Analysis using LLMs and LoRA Fine-Tuning

> A Capstone research project that builds a locally-hosted, privacy-preserving pipeline for automated network threat classification. It establishes a **6-Model Comparative Architecture** — pitting three Base LLMs (Zero-Shot) against their LoRA Fine-Tuned variants — using the UNSW-NB15 dataset.

---

## 📌 Project Overview

Modern Security Operations Centers (SOCs) are overwhelmed by thousands of daily alerts. While Large Language Models (LLMs) demonstrate strong reasoning capabilities, their out-of-the-box (Zero-Shot) performance on specialized network traffic analysis is often inaccurate and inconsistent.

This project answers the question:

> **"Can LoRA fine-tuning on a small local LLM outperform a large, un-trained model at classifying network threats?"**

The answer is proven empirically across three model families.

---

## 🏗️ Architecture

```
UNSW-NB15 Dataset (175k+ rows)
        │
        ▼
┌──────────────────────────────┐
│   prepare_lora.py            │  ← Converts CSV → conversational JSONL
│   80/20 Train/Test Split     │     train.jsonl (8,000) | test.jsonl (2,000)
└──────────────────────────────┘
        │
   ┌────┴──────────────────────────────────────┐
   │                                           │
   ▼                                           ▼
┌──────────────────┐               ┌──────────────────────┐
│  ZERO-SHOT PATH  │               │   LoRA FINE-TUNE PATH│
│                  │               │                      │
│  Llama 3.2 3B    │               │  train_lora.py       │
│  Gemma-2 2B      │               │  QLoRA (4-bit, r=16) │
│  Phi-4 Mini      │               │  SFTTrainer + bf16   │
│                  │               │                      │
│  evaluate_       │               │  evaluate_           │
│  zero_shot.py    │               │  metrics.py          │
└──────────────────┘               └──────────────────────┘
        │                                       │
        └──────────────────┬────────────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Master         │
                  │  Dashboard      │
                  │  Side-by-side   │
                  │  Comparison     │
                  └─────────────────┘
```

---

## 📊 Models Evaluated

| Model | Family | Parameters | Role |
|---|---|---|---|
| `microsoft/phi-4-mini-instruct` | Phi-4 | ~3.8B | Zero-Shot + LoRA Fine-Tuned |
| `unsloth/Llama-3.2-3B-Instruct` | Llama 3 | 3B | Zero-Shot + LoRA Fine-Tuned |
| `unsloth/gemma-2-2b-it` | Gemma-2 | 2B | Zero-Shot + LoRA Fine-Tuned |

---

## 📁 Repository Structure

```
├── LLM/                          # Zero-Shot evaluation scripts
│   ├── evaluate_zero_shot.py     # Ollama-based baseline evaluation
│   ├── evaluate_hf_phi4_zero_shot.py
│   ├── chat_all_base.py          # Interactive chat with base models
│   └── zero_shot_reports/        # Output JSON reports
│
├── Lora LLM/                     # LoRA fine-tuning pipeline
│   ├── phi4/                     # Phi-4 Mini LoRA scripts
│   ├── llama3/                   # Llama 3.2 LoRA scripts
│   ├── gemma2/                   # Gemma-2 LoRA scripts
│   ├── datasets/                 # train.jsonl / test.jsonl
│   └── metrics/                  # Evaluation output reports
│
├── Master Dashboard/             # Final comparison visualizations
├── demo_app.py                   # Streamlit demo (Zero-Shot)
├── demo_app_lora.py              # Streamlit demo (LoRA adapter)
├── hacker_lora_terminal.py       # Terminal-based LoRA inference UI
├── Master_Data_Aware_Chat.py     # Unified chat interface
└── Methodology_Report.md         # Full technical methodology
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM
- [Ollama](https://ollama.com/) installed (for Zero-Shot evaluation)

### 1. Setup Environment
```bash
git clone https://github.com/RS0907/Cybersec-log-analysis-using-LLM-and-Lora-LLM.git
cd Cybersec-log-analysis-using-LLM-and-Lora-LLM
python -m venv venv
.\venv\Scripts\activate
pip install torch transformers peft trl bitsandbytes datasets scikit-learn matplotlib tqdm
```

### 2. Prepare the Dataset
Download the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) CSV and place it in `Lora LLM/datasets/`.
```bash
cd "Lora LLM/phi4"
python prepare_lora.py
# Creates train.jsonl (8,000 rows) and test.jsonl (2,000 rows)
```

### 3. Fine-Tune with LoRA
```bash
python train_lora.py
# Trains Phi-4 Mini with QLoRA (r=16, 4-bit nf4) for 1 epoch
# Saves adapter to ./lora_unsw_final
```

### 4. Evaluate LoRA Accuracy
```bash
python evaluate_metrics.py
# Runs inference on 200 randomized hold-out samples
```

### 5. Run Zero-Shot Baseline
```bash
# Ensure Ollama is running: ollama serve
cd ../../LLM
python evaluate_zero_shot.py
```

---

## 📈 Results (Phi-4 Mini)

| Metric | Zero-Shot | LoRA Fine-Tuned |
|---|---|---|
| Valid JSON Output Rate | ~60-75% | **100%** |
| Attack Type Accuracy | TBD | **67.5%** |
| Severity Accuracy | TBD | **86.5%** |

> LoRA fine-tuning achieved **100% valid JSON formatting** — a critical requirement for SOC automation pipelines — compared to inconsistent outputs from the Zero-Shot baseline.

---

## ⚙️ LoRA Configuration

| Hyperparameter | Value |
|---|---|
| Rank (`r`) | 16 |
| LoRA Alpha | 32 |
| Target Modules | `all-linear` |
| Quantization | 4-bit NF4 (QLoRA) |
| Training Samples | 8,000 |
| Epochs | 1 |
| Batch Size | 2 (+ 4 grad accumulation steps) |
| Optimizer | `paged_adamw_32bit` |
| Precision | BFloat16 |

---

## 📋 Dataset

**UNSW-NB15** — Created by the Australian Centre for Cyber Security (ACCS).
- 175,341 records of real and synthetic network traffic
- 9 attack categories: *Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms*
- Features translated from tabular CSV → natural language for LLM consumption

---

## 🔬 Methodology

See [Methodology_Report.md](./Methodology_Report.md) for a full technical breakdown of the data preprocessing, QLoRA training process, and evaluation design.

---

## 👤 Author

**Rahul** — Capstone Project, 2026
