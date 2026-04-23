import pandas as pd
import json
import time
import random
import ollama
from pathlib import Path

from ollama_client import analyze_logs_with_ollama
from json_utils import safe_extract_json
from file_utils import save_json

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR.parent / "Lora" / "unsw_nb15.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "unsw_report.json"

SYSTEM_PROMPT = """
You are a cybersecurity SOC analyst specializing in network traffic flow analysis.

STRICT RULES:
- Output ONLY a valid JSON object.
- No explanations.
- No markdown.
- No extra text.

The JSON object MUST contain exactly these keys:
"attack_type" (e.g., Normal, Fuzzers, DoS, Exploits, Reconnaissance, Generic, etc)
"severity" (e.g., HIGH, MEDIUM, LOW)
"recommended_action" (A short sentence describing what to do)
"""

print(f"Loading dataset from {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Sample 200 random rows so the LLM processes it in minutes rather than days
df_sample = df.sample(n=min(200, len(df)), random_state=42)

all_events = []
MAX_RETRIES = 3

print("Sending network flows for analysis via LLAMA...")

for idx, (_, row) in enumerate(df_sample.iterrows(), start=1):
    print(f" Processing flow {idx}/200...")
    
    # Convert CSV row to the natural language prompt format matching the project
    prepared_flow = (
        f"Network traffic using {row.get('proto', 'Unknown')} protocol, "
        f"service {row.get('service', 'Unknown')}, state {row.get('state', 'Unknown')}, "
        f"source bytes {row.get('sbytes', 0)}, destination bytes {row.get('dbytes', 0)}, "
        f"rate {row.get('rate', 0)}."
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            ignore_cache = (attempt > 0)
            response = analyze_logs_with_ollama(SYSTEM_PROMPT, prepared_flow, ignore_cache=ignore_cache)
            
            try:
                parsed = safe_extract_json(response)
            except ValueError:
                if attempt < MAX_RETRIES - 1:
                    print(f"    [Retry] No generic JSON Object found. Requesting fresh generation...")
                    continue
                else:
                    print(f" No JSON found for flow {idx}, skipping.")
                    break
                    
            # Ensure the LLAMA response gave us a JSON dictionary
            if not isinstance(parsed, dict) or "attack_type" not in parsed:
                if attempt < MAX_RETRIES - 1:
                    print(f"    [Retry] Invalid JSON structure. Requesting fresh generation...")
                    continue
                else:
                    print(f" Invalid JSON structure for flow {idx}, skipping.")
                    break
                    
            # Synthesize an event record
            # The UNSW dataset has source/dest IP in older raw PCAP versions, but the CSV flow drops it.
            # We generate dummy IPs here just so the Power BI dashboard format continues working smoothly.
            event = {
                "timestamp": time.strftime("%b %d %H:%M:%S"),
                "source_ip": f"192.168.1.{random.randint(2, 254)}",
                "destination_ip": "SERVER_IP",
                "attack_type": parsed.get("attack_type", "Unknown"),
                "severity": parsed.get("severity", "LOW"),
                "recommended_action": parsed.get("recommended_action", "Monitor traffic")
            }
            all_events.append(event)
            break
            
        except json.JSONDecodeError as e:
             if attempt < MAX_RETRIES - 1:
                 print(f"    [Retry] JSON parse error: {e}. Requesting fresh generation...")
                 continue
             else:
                 print(f"!.Failed to process flow {idx} after {MAX_RETRIES} attempts.")
                 break
        except Exception as e:
            print(f"!.Failed to process flow {idx}: {e}")
            break

# ---------- FINAL REPORT ----------
attack_counts = {}
for e in all_events:
    atk = e["attack_type"]
    attack_counts[atk] = attack_counts.get(atk, 0) + 1

# Calculate how many flows were actually malicious vs Normal
malicious_events = [e for e in all_events if e["attack_type"].lower() not in ["normal", "none", "unknown"]]

final_report = {
    "overview": {
        "total_attacks": len(malicious_events),
        "total_flows_analyzed": len(all_events),
        "unique_attack_types": len(set(e["attack_type"] for e in malicious_events)),
        "most_frequent_attack": max(attack_counts, key=attack_counts.get) if attack_counts else "None"
    },
    "attack_summary": [
        {
            "attack_type": k,
            "count": v,
            "severity": "HIGH" if "dos" in str(k).lower() or "exploit" in str(k).lower() else "MEDIUM"
        }
        for k, v in attack_counts.items()
    ],
    "events": all_events
}

print("\n========== UNSW-NB15 LLAMA ANALYSIS SUMMARY ==========\n")
overview = final_report["overview"]
print(f"Flows Analyzed          : {overview['total_flows_analyzed']}")
print(f"Total Attacks Detected  : {overview['total_attacks']}")
print(f"Unique Attack Types     : {overview['unique_attack_types']}")
print(f"Most Frequent Class     : {overview['most_frequent_attack']}")
print("\n======================================================\n")

save_json(OUTPUT_PATH, final_report)
print(f"Report saved to {OUTPUT_PATH} for Power BI.")

# ---------- INTERACTIVE Q&A PHASE ----------
print("\n" + "="*54)
print("   🤖 INTERACTIVE REPORT Q&A (Type 'exit' to quit) ")
print("="*54)

# Give Llama the report we just generated as its background knowledge
qa_context = (
    "You are a helpful Senior Security Analyst. "
    f"Here is the network security report you just generated based on UNSW-NB15 flows:\n"
    f"{json.dumps(final_report, indent=2)}\n\n"
    "Answer any questions the user has based solely on this report. Keep answers concise."
)

chat_history = [{"role": "system", "content": qa_context}]

while True:
    try:
        user_q = input("\n[Analyst Question]: ")
        if user_q.lower() in ["exit", "quit", "q"]:
            print("Exiting Q&A session. Great job today!")
            break
            
        if not user_q.strip():
            continue
            
        chat_history.append({"role": "user", "content": user_q})
        
        print("[Llama3]: ", end="", flush=True)
        # Use streaming so the user sees the answer type out in real-time
        response = ollama.chat(
            model="llama3.2", 
            messages=chat_history,
            stream=True 
        )
        
        full_answer = ""
        for chunk in response:
            text = chunk['message']['content']
            print(text, end="", flush=True)
            full_answer += text
        print()
        
        # Save the assistant's answer so it remembers the conversation history
        chat_history.append({"role": "assistant", "content": full_answer})
        
    except KeyboardInterrupt:
        print("\nExiting Q&A session. Great job today!")
        break
    except Exception as e:
        print(f"\n[Error communicating with Llama3]: {e}")
