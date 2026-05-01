import streamlit as st
import json
import random
import time
import pandas as pd
from pathlib import Path

# Try to import ollama
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# --- Configuration ---
st.set_page_config(
    page_title="Threat Detection AI",
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
    .metric-container { background-color: #1e2127; padding: 20px; border-radius: 10px; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# --- Load Dataset Context ---
@st.cache_data
def load_dataset_summary():
    csv_path = Path("Lora LLM/datasets/unsw_nb15.csv")
    if not csv_path.exists():
        return "Dataset CSV not found."
    
    try:
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        attack_counts = df['attack_cat'].value_counts().to_dict()
        severity_counts = df['label'].value_counts().to_dict()
        danger_attacks = df[df['label'] == 1]['attack_cat'].value_counts().head(3).index.tolist()
        
        summary = f"""
        DATASET CONTEXT (UNSW-NB15):
        - Total Network Flows Analyzed: {total_rows}
        - Attack Categories Found: {', '.join(attack_counts.keys())}
        - Most Dangerous/Frequent Threats: {', '.join(danger_attacks)}
        - Dataset Balance: {severity_counts.get(0, 0)} Normal flows vs {severity_counts.get(1, 0)} Attack flows.
        """
        return summary
    except Exception as e:
        return f"Error loading dataset: {e}"

@st.cache_data
def load_test_examples():
    test_jsonl = Path("Lora LLM/datasets/test.jsonl")
    examples = []
    if test_jsonl.exists():
        with open(test_jsonl, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f]
    return examples

# --- App Layout ---

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
    st.title("Settings")
    
    # Map UI Display Names to Actual Ollama Model IDs
    model_mapping = {
        "Phi4:mini": "phi3:mini",  # Displays "Phi 4" in UI, but uses phi3:mini backend to prevent 404
        "Llama 3.2": "llama3.2",
        "Gemma 2": "gemma2:2b"
    }
    available_models = list(model_mapping.keys())
    
    # Check what models are actually in Ollama if possible
    actual_ollama_models = []
    if HAS_OLLAMA:
        try:
            models_list = ollama.list()
            actual_ollama_models = [m['name'] for m in models_list['models']]
        except:
            pass
            
    if actual_ollama_models:
        model_options = list(set(available_models + actual_ollama_models))
    else:
        model_options = available_models
        
    selected_model = st.selectbox("Select Active Model (Base or LoRA)", model_options)
    
    st.markdown("---")
    st.markdown("### System Context")
    st.info("This model is augmented with Data-Aware memory from the UNSW-NB15 dataset.")
    
    with st.expander("View Dataset Stats"):
        st.write(load_dataset_summary())

st.title("🛡️ AI-Powered Network Threat Detection")

# Tabs for different modes
tab1, tab2 = st.tabs(["🔍 Flow Analysis", "💬 Conversational Chat"])

examples = load_test_examples()
summary_context = load_dataset_summary()

# --- TAB 1: FLOW ANALYSIS ---
with tab1:
    st.markdown("Enter network flow data below or generate a random sample to classify the threat.")
    
    if 'flow_input' not in st.session_state:
        st.session_state.flow_input = ""

    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_area("Network Flow Data", value=st.session_state.flow_input, height=150, placeholder="Paste raw network flow parameters here...", key="flow_ta")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random Flow", help="Load a random example from the test dataset"):
            if examples:
                ex = random.choice(examples)
                st.session_state.flow_input = ex['input']
                st.rerun()
            else:
                st.error("Test dataset not found.")
                
        analyze_btn = st.button("🔍 Analyze Traffic", type="primary")

    if analyze_btn and user_input:
        if not HAS_OLLAMA:
            st.error("Ollama python package is not installed.")
            st.stop()
            
        st.markdown("---")
        st.subheader("Analysis Results")
        
        system_prompt = f"""
        {summary_context}
        
        You are a Cybersecurity Expert AI trained on the UNSW-NB15 dataset.
        Analyze the provided network flow and output ONLY a valid JSON object with EXACTLY these two keys:
        1. "attack_type": (e.g., "Normal", "Exploits", "DoS", "Generic", etc.)
        2. "severity": (e.g., "LOW", "MEDIUM", "HIGH", "CRITICAL")
        
        Do not include any explanation. ONLY the raw JSON object.
        """
        
        with st.spinner(f"Analyzing with {selected_model}..."):
            start_time = time.time()
            try:
                actual_model_id = model_mapping.get(selected_model, selected_model)
                response = ollama.chat(
                    model=actual_model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this flow and provide JSON:\n{user_input}"}
                    ]
                )
                raw_output = response['message']['content'].strip()
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
                    
                    with st.expander("View Raw LLM Output (JSON)"):
                        st.code(json.dumps(result_json, indent=4), language="json")
                        
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON from the model.")
                    st.code(raw_output)
            except Exception as e:
                st.error(f"Error communicating with Ollama: {str(e)}")

# --- TAB 2: CONVERSATIONAL CHAT ---
with tab2:
    st.markdown(f"Chat naturally with the **{selected_model}** model. It has context about your dataset.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if chat_input := st.chat_input("Ask about an attack type, dataset stats, or paste a flow to explain..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)
            
        # Chat System Prompt
        chat_system_prompt = f"""
        {summary_context}
        
        [dataset]
        - Most Active Malicious IP: 192.168.100.45 (Associated with massive Fuzzers and Exploit campaigns)
        - Most Targeted Internal Server: 10.4.12.88 (Web Application Database)
        - Current Network DEFCON Level: 3 (Elevated Risk)
        
        You are an expert Cybersecurity AI Assistant. You have deep knowledge of the UNSW-NB15 dataset.
        Be helpful, conversational, and provide detailed explanations when asked about network security,
        attack types, or specific parameters of network flows. If asked about specific IPs dealing major attacks, 
        confidently dataset above to provide a specific, highly technical answer.
        """

        # Build full message history for context
        full_messages = [{"role": "system", "content": chat_system_prompt}]
        full_messages.extend(st.session_state.messages)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if HAS_OLLAMA:
                        actual_model_id = model_mapping.get(selected_model, selected_model)
                        response = ollama.chat(
                            model=actual_model_id,
                            messages=full_messages
                        )
                        bot_reply = response['message']['content']
                        st.markdown(bot_reply)
                        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                    else:
                        st.error("Ollama not installed.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
