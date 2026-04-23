import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

base_model_name = "unsloth/gemma-2-2b-it"
BASE_DIR = Path(__file__).parent
lora_dir = str(BASE_DIR / "lora_gemma_v3_final")

if not os.path.exists(lora_dir):
    print(f"Error: {lora_dir} does not exist yet. Please run train_gemma.py first.")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading Gemma base model in 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

print(f"Loading LoRA adapter from {lora_dir}...")
model = PeftModel.from_pretrained(base_model, lora_dir)
model.eval()

print("\n" + "="*50)
print(" 🚀 INTERACTIVE GEMMA-2 LORA CHAT LAUNCHED 🚀")
print(" Type a network flow to test it, or type 'exit' to quit.")
print("="*50)

while True:
    raw_network_flow = input("\n[Paste Network Flow]: ").strip()
    
    if raw_network_flow.lower() in ['exit', 'quit']:
        print("Ending interactive session. Great job on the training!")
        break
        
    if not raw_network_flow:
        continue

    # Format exactly how the Gemma model was instructed
    messages = [
        {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{raw_network_flow}"}
    ]

    # Apply Gemma specific chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\n[LoRA Prediction]:\n{prediction}")
