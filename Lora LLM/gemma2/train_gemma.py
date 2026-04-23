import os
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_name = "unsloth/gemma-2-2b-it"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "datasets"

print("1. Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print("2. Configuring 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("3. Loading Base Gemma Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
model.config.use_cache = False

print("4. Applying LoRA specifically for Gemma Architecture...")
lora_config = LoraConfig(
    r=32,           # Increased from 16 → more capacity for 10-class classification
    lora_alpha=64,  # Keep alpha = 2x rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print(f"5. Loading Datasets from {DATA_DIR}...")
train_file = str(DATA_DIR / "train.jsonl")
test_file = str(DATA_DIR / "test.jsonl")
raw_dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})

def convert_to_chat(example):
    return {
        "messages": [
            {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{example['input']}"},
            {"role": "model", "content": example['output']}  # Gemma technically uses 'model' instead of 'assistant' in standard chat templates
        ]
    }

dataset = raw_dataset.map(convert_to_chat)

print("6. Setting up SFT Trainer...")
training_args = SFTConfig(
    output_dir="./lora_gemma_v3",
    per_device_train_batch_size=1,  # LAPTOP SAFE MODE!
    gradient_accumulation_steps=8,  # Effective batch = 8
    num_train_epochs=5,             # Increased from 3 → more learning on 8000 samples
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,              # Gentle warmup prevents early overfitting
    weight_decay=0.01,              # Regularization
    report_to="none",
    max_length=512,                 # Increased from 256 → capture full network traffic context
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    peft_config=lora_config,
)

print("7. Starting Training...")
trainer.train()

print("8. Saving Model...")
trainer.save_model("./lora_gemma_v3_final")
tokenizer.save_pretrained("./lora_gemma_v3_final")

print("✅ Gemma LoRA training complete and saved to ./lora_gemma_v3_final")
