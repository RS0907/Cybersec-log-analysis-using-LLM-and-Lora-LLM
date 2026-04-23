import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_name = "microsoft/phi-4-mini-instruct"

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

print("3. Loading Base Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
model.config.use_cache = False

print("4. Applying LoRA...")
lora_config = LoraConfig(
    r=32,           # Increased from 16 → more capacity for 10-class classification
    lora_alpha=64,  # Keep alpha = 2x rank
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "datasets"
print(f"5. Loading Datasets from {DATA_DIR}...")
raw_dataset = load_dataset("json", data_files={"train": str(DATA_DIR / "train.jsonl"), "test": str(DATA_DIR / "test.jsonl")})

# Modern TRL leverages the model's native chat template to automatically mask user prompts.
# This prevents the deprecated DataCollator import error and gives better accuracy!
def convert_to_chat(example):
    return {
        "messages": [
            {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
    }

dataset = raw_dataset.map(convert_to_chat)

print("6. Setting up SFT Trainer...")
training_args = SFTConfig(
    output_dir="./lora_unsw_v3",
    per_device_train_batch_size=1,  # LAPTOP SAFE MODE
    gradient_accumulation_steps=8,  # Effective batch = 8
    num_train_epochs=5,             # Increased from 3 → more learning on 8000 samples
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
    fp16=False,
    optim="paged_adamw_32bit",
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
trainer.save_model("./lora_unsw_v3_final")
tokenizer.save_pretrained("./lora_unsw_v3_final")

print("✅ LoRA training complete and saved to ./lora_unsw_v3_final")