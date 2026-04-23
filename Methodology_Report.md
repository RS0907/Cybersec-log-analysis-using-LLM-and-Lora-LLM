# Executive Summary: 6-Model Comparative Analysis for Network Threat Detection

## High-Level Overview
Modern Security Operations Centers (SOCs) are burdened by thousands of daily alerts, requiring automated, intelligent triage. While Large Language Models (LLMs) demonstrate profound reasoning capabilities, their out-of-the-box (Zero-Shot) performance on highly specialized, tabular network traffic analysis is often misaligned and computationally expensive. 

This Capstone project engineers a locally hosted, privacy-preserving pipeline that dynamically translates standard network flow metrics (UNSW-NB15 dataset) into natural language logs. It establishes a rigorous **6-Model Comparative Architecture**, pitting three distinct Base LLMs (Zero-Shot) against their Parameter-Efficient Fine-Tuned (PEFT/LoRA) variants. The objective is to empirically prove how fine-tuning dramatically increases attack classification accuracy, severity scaling, and strict JSON output formatting while operating efficiently on constrained consumer hardware.

---

# Detailed Methodology

## 1. Data Selection and Conversational Preprocessing
* **Dataset:** The *UNSW-NB15* dataset was selected due to its modern reflection of contemporary network attacks (Fuzzers, DoS, Exploits).
* **Feature Translation:** LLMs do not natively parse dense CSV structures efficiently. We developed a pre-processing engine (`prepare_lora.py`) that maps structured network features (Protocol, State, Source/Destination Bytes, and Flow Rate) into conversational instructions:
  > *"Analyze this network traffic and format the threat evaluation as JSON: Network traffic using tcp protocol, service ftp, state FIN, source bytes 1234..."*
* **Ground Truth:** The targets (`attack_cat` and severity) were encoded into strict, deterministic JSON structures necessary for programmatic SOC integrations.

## 2. Zero-Shot Baseline Evaluation Phase
To establish a control baseline, we built an orchestration pipeline (`analyze_unsw.py` and `evaluate_zero_shot.py`) relying on the native capabilities of off-the-shelf instruction models:
- Flow strings are piped blindly into the LLMs using rigorous Chat Templates.
- We measure: **Attack Type Accuracy**, **Severity Accuracy**, and the **Valid JSON Format Rate** (how often the model "hallucinates" outside of the requested JSON brackets).

## 3. High-Efficiency LoRA Fine-Tuning (QLoRA) Phase
Because fine-tuning billions of parameters requires massive industrial computing clusters, we implemented **QLoRA (Quantized Low-Rank Adaptation)** to make training viable on local hardware. 
- **4-Bit Quantization:** All base models are loaded in 4-bit `nf4` precision using `BitsAndBytes`, reducing VRAM consumption by up to 75%.
- **Adapter Injection:** Instead of modifying the entire multi-gigabyte neural network, we inject tiny, trainable low-rank matrices (`r=16`) specifically into the Attention Projection layers (e.g., `q_proj`, `v_proj`).
- **Gradient Accumulation:** To prevent Out-Of-Memory (OOM) crashes, we utilized micro-batching (batch sizes of 1 or 2) mathematically compounded over multiple gradient accumulation steps. 

## 4. The Three-Tier Model Selection
To provide a comprehensive analysis, models were selected across three distinct algorithmic families and weight classes:

1. **Phi-4 Mini Instruct (`microsoft/phi-4-mini-instruct`)** 
   * *Architecture:* Microsoft's highly-filtered, synthetic-data trained powerhouse.
   * *Purpose:* Tests if models trained specifically on code/logic excel natively at JSON formatting and rigid flow analysis.
2. **Llama-3.2 3B (`unsloth/Llama-3.2-3B-Instruct`)**
   * *Architecture:* Meta's newly released edge-optimized dense model. 
   * *Purpose:* Serves as the "middle-weight" industry standard, balancing deep reasoning parameters with compute efficiency.
3. **Gemma-2 2B (`unsloth/gemma-2-2b-it`)**
   * *Architecture:* Google's DeepMind-engineered lightweight architecture.
   * *Purpose:* Tests the absolute lower boundary of parameter size (2 Billion) to see if LoRA fine-tuning can force a tiny model to outperform massive, un-trained Zero-Shot counterparts.

## 5. Automated Evaluation and Master Dashboarding
Post-training, we execute dedicated inference evaluations (`evaluate_metrics.py`) designed to pass exactly 200 consistent, seeded network flows through the fused LoRA adapters. 

A unified orchestration script (`plot_dashboard.py`) dynamically reads the HuggingFace `trainer_state.json` histories and the evaluation JSONs. It renders a master, two-panel graphical dashboard encompassing all 6 models:
1. **Left Panel:** Plots the Cross-Entropy Training Loss decay, validating algorithmic convergence.
2. **Right Panel:** Visually compares Zero-Shot vs LoRA accuracies side-by-side, categorically proving the necessity and efficacy of the Fine-Tuning phase.
