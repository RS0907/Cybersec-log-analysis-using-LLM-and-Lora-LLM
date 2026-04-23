import json
import pandas as pd

def load_logs(file_path, log_type=None):
    logs = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract key=value pairs safely
            fields = {}
            for token in line.split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key.upper()] = value

            log_entry = {
                "timestamp": " ".join(line.split()[:2]),
                "source_ip": fields.get("SRC", "unknown"),
                "destination_ip": fields.get("DST", "unknown"),
                "protocol": fields.get("PROTO", "unknown"),
                "source_port": fields.get("SPT", "unknown"),
                "destination_port": fields.get("DPT", "unknown"),
                "alert_type": fields.get("ALERT", "unknown"),
                "severity": fields.get("SEVERITY", "unknown"),
                "raw_log": line
            }

            logs.append(log_entry)

    return logs
