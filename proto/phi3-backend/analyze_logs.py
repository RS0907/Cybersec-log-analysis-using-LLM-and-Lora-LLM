from pathlib import Path
import json

from Chunk_log import read_log_in_chunks
from ollama_client import analyze_logs_with_ollama
from json_utils import extract_json, normalize_events
from file_utils import save_json

BASE_DIR = Path(__file__).parent
LOGS_PATH = BASE_DIR / "logs/SSH_samp.log"
OUTPUT_PATH = BASE_DIR / "outputs/report1.json"

SYSTEM_PROMPT = """
You are a cybersecurity SOC analyst.

STRICT RULES:
- Output ONLY a valid JSON array
- No explanations
- No markdown
- No extra text

Each JSON object MUST contain:
timestamp, source_ip, attack_type, severity, recommended_action

Attack type is always: SSH Brute Force
Severity: HIGH if repeated failures, MEDIUM otherwise
destination_ip must NOT be included
"""

def preprocess(lines):
    tagged = []
    for line in lines:
        if "Failed password" in line:
            tagged.append("ATTACK BRUTE_FORCE " + line)
        elif "Accepted password" in line:
            tagged.append("LOGIN_SUCCESS " + line)
        else:
            tagged.append("OTHER " + line)
    return "\n".join(tagged)

print(f"Loading logs from {LOGS_PATH}")
print("Sending logs for analysis...")

all_events = []

for idx, chunk in enumerate(read_log_in_chunks(LOGS_PATH, chunk_size=100), start=1):
    print(f" Processing chunk {idx}")

    try:
        prepared = preprocess(chunk)
        response = analyze_logs_with_ollama(SYSTEM_PROMPT, prepared)

        json_text = extract_json(response)
        if not json_text:
            print(f" No JSON found in chunk {idx}, skipping")
            continue

        parsed = json.loads(json_text)
        if not isinstance(parsed, list):
            print(f" Invalid JSON structure in chunk {idx}")
            continue

        events = normalize_events(parsed)
        all_events.extend(events)

    except Exception as e:
        print(f"!.Failed to process chunk {idx}: {e}")

# ---------- FINAL REPORT ----------

attack_counts = {}
for e in all_events:
    atk = e["attack_type"]
    attack_counts[atk] = attack_counts.get(atk, 0) + 1

final_report = {
    "overview": {
        "total_attacks": len(all_events),
        "unique_attack_types": len(attack_counts),
        "most_frequent_attack": max(attack_counts, key=attack_counts.get)
        if attack_counts else "None"
    },
    "attack_summary": [
        {
            "attack_type": k,
            "count": v,
            "owasp_category": "Broken Authentication",
            "severity": "HIGH"
        }
        for k, v in attack_counts.items()
    ],
    "events": all_events
}

print("\n========== SSH ATTACK ANALYSIS SUMMARY ==========\n")

overview = final_report["overview"]
events = final_report["events"]

print(f"Total Attacks           : {overview['total_attacks']}")
print(f"Unique Attack Types     : {overview['unique_attack_types']}")
print(f"Most Frequent Attack    : {overview['most_frequent_attack']}")

# ---- Severity distribution ----
severity_count = {}
for e in events:
    sev = e["severity"]
    severity_count[sev] = severity_count.get(sev, 0) + 1

print("\nSeverity Distribution:")
for sev, count in severity_count.items():
    print(f"  {sev:<10} : {count}")

# ---- Top attacker IPs ----
ip_count = {}
for e in events:
    ip = e["source_ip"]
    ip_count[ip] = ip_count.get(ip, 0) + 1

top_ips = sorted(ip_count.items(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop Attacking IPs:")
for ip, count in top_ips:
    print(f"  {ip:<15} : {count} attempts")

# ---- Most critical attack ----
high_sev_events = [e for e in events if e["severity"] == "HIGH"]

if high_sev_events:
    # Count attempts per IP among HIGH severity attacks
    high_ip_count = {}
    for e in high_sev_events:
        ip = e["source_ip"]
        high_ip_count[ip] = high_ip_count.get(ip, 0) + 1

    # Find IP with max attempts
    critical_ip = max(high_ip_count, key=high_ip_count.get)

    # Get latest event from that IP
    critical_events = [
        e for e in high_sev_events if e["source_ip"] == critical_ip
    ]
    critical_event = sorted(
        critical_events, key=lambda x: x["timestamp"], reverse=True
    )[0]

    print("\nMost Critical Attack Detected:")
    print(f"  Attack Type     : {critical_event['attack_type']}")
    print(f"  Severity        : {critical_event['severity']}")
    print(f"  Source IP       : {critical_event['source_ip']}")
    print(f"  Destination IP : {critical_event['destination_ip']}")
    print(f"  Timestamp       : {critical_event['timestamp']}")
    print(f"  Attempts        : {high_ip_count[critical_ip]}")
    print(f"  Action          : {critical_event['recommended_action']}")
else:
    print("\nNo HIGH severity attacks detected.")


print("\n===============================================\n")


save_json(OUTPUT_PATH, final_report)
print("PHI3 model used for this analysis.")
print("Analysis complete. Report saved for Power BI.")
