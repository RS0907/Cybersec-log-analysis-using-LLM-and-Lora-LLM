import os
import time
import json
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import track

# Initialize Rich Console for Hacker Aesthetics
console = Console()

# Define the models and paths
LORA_MODELS = {
    "1": {
        "name": "Phi-4 (LoRA Optimized)",
        "base": "microsoft/phi-4-mini-instruct",
        "lora_dir": "Lora LLM/phi4/lora_unsw_v3_final"
    },
    "2": {
        "name": "Llama-3.2 (LoRA Optimized)",
        "base": "unsloth/Llama-3.2-3B-Instruct",
        "lora_dir": "Lora LLM/llama3/lora_llama_v3_final"
    },
    "3": {
        "name": "Gemma-2 (LoRA Optimized)",
        "base": "unsloth/gemma-2-2b-it",
        "lora_dir": "Lora LLM/gemma2/lora_gemma_v3_final"
    }
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    banner = """
    ███████╗ ██████╗  ██████╗    ██████╗ ███████╗████████╗███████╗ ██████╗████████╗
    ██╔════╝██╔═══██╗██╔════╝    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
    ███████╗██║   ██║██║         ██║  ██║█████╗     ██║   █████╗  ██║        ██║   
    ╚════██║██║   ██║██║         ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   
    ███████║╚██████╔╝╚██████╗    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   
    ╚══════╝ ╚═════╝  ╚═════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   
             [ LoRA Fine-Tuned Network Anomaly Detection System v1.0 ]
    """
    console.print(Panel.fit(Text(banner, style="bold cyan"), border_style="blue"))

def load_dataset():
    test_jsonl = Path("Lora LLM/datasets/test.jsonl")
    examples = []
    if test_jsonl.exists():
        with open(test_jsonl, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f]
    return examples

def load_hf_model(model_choice):
    model_info = LORA_MODELS[model_choice]
    base_model_name = model_info["base"]
    lora_dir = model_info["lora_dir"]
    
    console.print(f"\n[bold yellow][*] Initializing GPU VRAM for {model_info['name']}...[/]")
    time.sleep(1)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    with console.status("[bold green]Mounting Base Weights into Neural Core...[/]", spinner="bouncingBar"):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
    
    with console.status("[bold red]Injecting LoRA Threat-Detection Adapters...[/]", spinner="bouncingBar"):
        model = PeftModel.from_pretrained(base_model, lora_dir)
        model.eval()
        
    console.print("[bold green][+] Neural network online and armed.[/]\n")
    return tokenizer, model

def main():
    clear_screen()
    print_banner()
    
    examples = load_dataset()
    if not examples:
        console.print("[bold red][!] Warning: Could not locate test.jsonl dataset.[/]")
        
    console.print("\n[bold white]Available LoRA Architectures:[/]")
    for key, val in LORA_MODELS.items():
        console.print(f"  [[bold cyan]{key}[/]] {val['name']}")
        
    choice = input("\nroot@soc-system:~# Select model index (1-3): ").strip()
    if choice not in LORA_MODELS:
        console.print("[bold red][!] Invalid selection. Aborting.[/]")
        return
        
    try:
        tokenizer, model = load_hf_model(choice)
    except Exception as e:
        console.print(f"\n[bold red][!] FATAL SYSTEM ERROR: {e}[/]")
        console.print("[yellow]Hint: Make sure Ollama is closed so it isn't hogging your 6GB GPU VRAM![/]")
        return
        
    console.print(Panel(f"CONNECTION ESTABLISHED: {LORA_MODELS[choice]['name']} \nCommands: 'r' (Random Flow), 'clear' (Clear Screen), 'exit' (Terminate)", style="bold green"))
    
    while True:
        try:
            user_input = console.input("\n[bold green]soc_analyst@threat-intel[/] [bold blue]~[/]$ ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("\n[bold red]Terminating secure connection...[/]")
                break
            elif user_input.lower() == 'clear':
                clear_screen()
                print_banner()
                continue
            elif user_input.lower() == 'r':
                if examples:
                    ex = random.choice(examples)
                    user_input = ex['input']
                    console.print(Panel(user_input, title="[Intercepted Network Flow]", border_style="yellow"))
                else:
                    console.print("[bold red][!] No dataset available.[/]")
                    continue
            
            if not user_input.strip():
                continue
                
            messages = [
                {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{user_input}"}
            ]
            
            with console.status("[bold magenta]Scanning flow vectors via LoRA matrix...[/]", spinner="dots"):
                start_time = time.time()
                try:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    # Fallback for Gemma/Llama if needed
                    prompt = f"<|user|>\nAnalyze this network traffic and format the threat evaluation as JSON:\n{user_input}\n<|assistant|>\n"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=150,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] if "Llama" in LORA_MODELS[choice]["name"] else tokenizer.eos_token_id,
                        temperature=0.1,
                        do_sample=False,
                        repetition_penalty=1.1
                    )
                
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                inf_time = time.time() - start_time
                
            # Formatting the output nicely
            if raw_output.startswith("```json"): raw_output = raw_output[7:]
            if raw_output.startswith("```"): raw_output = raw_output[3:]
            if raw_output.endswith("```"): raw_output = raw_output[:-3]
            raw_output = raw_output.strip()
            
            try:
                result_json = json.loads(raw_output)
                attack_type = result_json.get('attack_type', 'UNKNOWN')
                severity = result_json.get('severity', 'UNKNOWN')
                
                if severity.upper() in ["HIGH", "CRITICAL"]:
                    color = "red"
                else:
                    color = "green"
                    
                output_panel = f"""
[bold white]THREAT ASSESSMENT:[/]
Attack Vector  : [bold {color}]{attack_type}[/]
Severity Level : [bold {color} blink]{severity}[/]
--------------------------------
[grey50]Response Time  : {inf_time:.2f}s[/]
                """
                console.print(Panel(output_panel.strip(), border_style=color, title=f"[{LORA_MODELS[choice]['name']} Analysis]"))
                
            except Exception:
                console.print(f"[bold red]Raw Output:[/] \n{raw_output}")
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
