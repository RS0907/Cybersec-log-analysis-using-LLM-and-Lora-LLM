import os
import streamlit as st
import json
import random
import time
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
st.set_page_config(
    page_title="LoRA Threat Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .severity-high { color: #ff4b4b; font-weight: bold; font-size: 24px; }
    .severity-low { color: #00cc96; font-weight: bold; font-size: 24px; }
    .metric-container { background-color: #1e2127; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 20px;}
    .loading-text { font-size: 18px; color: #f39c12; }
    </style>
""", unsafe_allow_html=True)

# Define the models and paths
LORA_MODELS = {
    "Phi-4 (LoRA Fine-Tuned)": {
        "base": "microsoft/phi-4-mini-instruct",
        "lora_dir": "Lora LLM/phi4/lora_unsw_v3_final"
    },
    "Llama-3.2 (LoRA Fine-Tuned)": {
        "base": "unsloth/Llama-3.2-3B-Instruct",
        "lora_dir": "Lora LLM/llama3/lora_llama_v3_final"
    },
    "Gemma-2 (LoRA Fine-Tuned)": {
        "base": "unsloth/gemma-2-2b-it",
        "lora_dir": "Lora LLM/gemma2/lora_gemma_v3_final"
    }
}

# --- Model Loading (Manual Memory Management to prevent VRAM overflow) ---
def load_hf_model(model_choice):
    model_info = LORA_MODELS[model_choice]
    base_model_name = model_info["base"]
    lora_dir = model_info["lora_dir"]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()
    return tokenizer, model

# --- Load Dataset Context ---
@st.cache_data
def load_dataset_summary():
    csv_path = Path("Lora LLM/datasets/unsw_nb15.csv")
    if not csv_path.exists(): return "Dataset CSV not found."
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    attack_counts = df['attack_cat'].value_counts().to_dict()
    severity_counts = df['label'].value_counts().to_dict()
    danger_attacks = df[df['label'] == 1]['attack_cat'].value_counts().head(3).index.tolist()
    return f"""DATASET CONTEXT (UNSW-NB15):
- Total Flows: {total_rows}
- Attack Categories: {', '.join(attack_counts.keys())}
- Most Dangerous: {', '.join(danger_attacks)}"""

@st.cache_data
def load_test_examples():
    test_jsonl = Path("Lora LLM/datasets/test.jsonl")
    examples = []
    if test_jsonl.exists():
        with open(test_jsonl, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f]
    return examples

# --- App Layout ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
    st.title("Model Settings")
    
    # Allow user to switch between their 3 LoRA models
    selected_model_name = st.selectbox("Select Active LoRA Model", list(LORA_MODELS.keys()))
    
    # Manage VRAM Memory aggressively
    if "current_model_name" not in st.session_state:
        st.session_state.current_model_name = None

    if st.session_state.current_model_name != selected_model_name:
        st.markdown("### Memory Management")
        with st.spinner("Flushing previous model from VRAM..."):
            if "hf_model" in st.session_state:
                del st.session_state["hf_model"]
                del st.session_state["hf_tokenizer"]
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2) # Give Windows a second to reclaim memory
            
        with st.spinner(f"Loading {selected_model_name} into VRAM..."):
            try:
                tokenizer, model = load_hf_model(selected_model_name)
                st.session_state["hf_tokenizer"] = tokenizer
                st.session_state["hf_model"] = model
                st.session_state.current_model_name = selected_model_name
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()
                
    st.markdown("---")
    st.title("LoRA Hardware Status")
    if "hf_model" in st.session_state:
        st.success(f"✅ {st.session_state.current_model_name} Loaded via PyTorch/PEFT")
        tokenizer = st.session_state["hf_tokenizer"]
        model = st.session_state["hf_model"]
            
    st.markdown("---")
    st.markdown("### System Context")
    st.info("This interface runs directly on your custom HuggingFace PyTorch LoRA weights, proving the effectiveness of your fine-tuning.")
    
    with st.expander("View Dataset Stats"):
        st.write(load_dataset_summary())

st.title(f"🛡️ {selected_model_name} Interface")

tab1, tab2 = st.tabs(["🔍 Strict Flow Analysis", "💬 Conversational Chat"])

examples = load_test_examples()

# --- TAB 1: FLOW ANALYSIS ---
with tab1:
    st.markdown("Enter network flow data below or generate a random sample to test your **actual fine-tuned LoRA model**.")
    
    if 'hf_flow_input' not in st.session_state:
        st.session_state.hf_flow_input = ""

    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_area("Network Flow Data", value=st.session_state.hf_flow_input, height=150, placeholder="Paste raw network flow parameters here...", key="hf_flow_ta")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random Flow", help="Load a random example from the test dataset", key="btn_rand"):
            if examples:
                ex = random.choice(examples)
                st.session_state.hf_flow_input = ex['input']
                st.rerun()
            else:
                st.error("Test dataset not found.")
                
        analyze_btn = st.button("🔍 Analyze Traffic", type="primary", key="btn_analyze")

    if analyze_btn and user_input:
        st.markdown("---")
        st.subheader("Analysis Results")
        
        with st.spinner("Running Inference through LoRA Adapter..."):
            start_time = time.time()
            try:
                messages = [
                    {"role": "user", "content": f"Analyze this network traffic and format the threat evaluation as JSON:\n{user_input}"}
                ]
                # Fallback template if Llama/Gemma fails on tokenizer.apply_chat_template
                try:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    prompt = f"<|user|>\nAnalyze this network traffic and format the threat evaluation as JSON:\n{user_input}\n<|assistant|>\n"
                    
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=150,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] if "Llama" in selected_model_name else tokenizer.eos_token_id,
                        temperature=0.1,
                        do_sample=False,
                        repetition_penalty=1.1
                    )
                
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                inference_time = time.time() - start_time
                
                if raw_output.startswith("```json"): raw_output = raw_output[7:]
                if raw_output.startswith("```"): raw_output = raw_output[3:]
                if raw_output.endswith("```"): raw_output = raw_output[:-3]
                raw_output = raw_output.strip()
                
                try:
                    result_json = json.loads(raw_output)
                    attack_type = result_json.get('attack_type', 'UNKNOWN')
                    severity = result_json.get('severity', 'UNKNOWN')
                    
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1: st.metric("Attack Category", attack_type)
                    with mcol2:
                        if severity.upper() in ["HIGH", "CRITICAL"]:
                            st.markdown(f"**Severity:** <br><span class='severity-high'>🚨 {severity.upper()}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**Severity:** <br><span class='severity-low'>✅ {severity.upper()}</span>", unsafe_allow_html=True)
                    with mcol3: st.metric("Inference Time", f"{inference_time:.2f}s")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    with st.expander("View Raw LoRA Output (JSON)"):
                        st.code(json.dumps(result_json, indent=4), language="json")
                        
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON from the model.")
                    st.code(raw_output)
            except Exception as e:
                st.error(f"Inference Error: {str(e)}")

# --- TAB 2: CONVERSATIONAL CHAT ---
with tab2:
    st.markdown(f"Chat naturally with your **{selected_model_name}** directly through PyTorch/HuggingFace.")
    
    if "hf_messages" not in st.session_state:
        st.session_state.hf_messages = []

    for message in st.session_state.hf_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if chat_input := st.chat_input("Ask your LoRA model a question..."):
        st.session_state.hf_messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                try:
                    try:
                        prompt = tokenizer.apply_chat_template(st.session_state.hf_messages, tokenize=False, add_generation_prompt=True)
                    except:
                        # Fallback for simpler models
                        prompt = ""
                        for msg in st.session_state.hf_messages:
                            role = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
                            prompt += f"{role}\n{msg['content']}\n"
                        prompt += "<|assistant|>\n"
                        
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=250,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] if "Llama" in selected_model_name else tokenizer.eos_token_id,
                            temperature=0.7
                        )
                    
                    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                    bot_reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    st.markdown(bot_reply)
                    st.session_state.hf_messages.append({"role": "assistant", "content": bot_reply})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
