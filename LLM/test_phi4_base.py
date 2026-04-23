import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_name = "microsoft/phi-4-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading RAW BASE model in 4-bit (No LoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
model.eval()

print("\n" + "="*50)
print(" 🤖 RAW PHI-4 BASE CHAT (ZERO-SHOT) 🤖")
print(" Compare this to your LoRA version!")
print("="*50)

while True:
    raw_input = input("\n[Paste Network Flow]: ").strip()
    
    if raw_input.lower() in ['exit', 'quit']:
        break
        
    if not raw_input:
        continue

    messages = [
        {"role": "system", "content": "You are a cybersecurity JSON generator. Output ONLY valid JSON with keys 'attack_type' and 'severity'."},
        {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{raw_input}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating zero-shot prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\n[Base Model Prediction]:\n{prediction}")
