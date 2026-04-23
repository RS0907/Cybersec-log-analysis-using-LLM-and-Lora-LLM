import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

base_model_name = "unsloth/Llama-3.2-3B-Instruct"
BASE_DIR = Path(__file__).parent
lora_dir = str(BASE_DIR / "lora_llama_final")

if not os.path.exists(lora_dir):
    print(f"Error: {lora_dir} does not exist yet. Please run train_llama.py first.")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading Llama-3 base model in 4-bit...")
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
print(" 🤖 OPEN LORA CHAT (NO JSON WRAPPER) 🤖")
print(" You are now speaking directly to your fine-tuned model.")
print(" Type 'exit' to quit.")
print("="*50)

# Keep track of history to make it a real chat!
conversation_history = []

while True:
    user_input = input("\n[You]: ").strip()
    
    if user_input.lower() in ['exit', 'quit']:
        print("Ending chat session.")
        break
        
    if not user_input:
        continue

    # Notice we removed the "Analyze this network traffic..." wrapper! 
    # We are just sending your raw text straight into the model now.
    conversation_history.append({"role": "user", "content": user_input})

    # Apply Llama-3 specific chat template
    prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("LoRA thinking...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7 # We turned the temperature up so it acts more conversational!
        )

    # Decode the response
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Save its response to memory so it remembers what you are talking about
    conversation_history.append({"role": "assistant", "content": prediction})

    print(f"\n[Llama-LoRA]:\n{prediction}")
