import json
from jsonschema import validate, ValidationError

# JSON Schema for validation
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {
            "type": "object",
            "properties": {
                "total_attacks": {"type": "number"},
                "unique_attack_types": {"type": "number"},
                "most_frequent_attack": {"type": "string"},
                "log_timeline": {"type": "string"}
            },
            "required": ["total_attacks", "unique_attack_types"]
        },
        "attack_summary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attack_type": {"type": "string"},
                    "count": {"type": "number"},
                    "owasp_category": {"type": "string"},
                    "owasp_description": {"type": "string"},
                    "severity": {"type": "string"}
                },
                "required": ["attack_type", "count", "owasp_category", "severity"]
            }
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "source_ip": {"type": "string"},
                    "destination_ip": {"type": "string"},
                    "attack_type": {"type": "string"},
                    "vulnerability": {"type": "string"},
                    "affected_service": {"type": "string"},
                    "severity": {"type": "string"},
                    "recommended_action": {"type": "string"}
                },
                "required": ["timestamp", "source_ip", "attack_type", "severity", "recommended_action"]
            }
        }
    },
    "required": ["overview", "attack_summary", "events"]
}

def safe_extract_json(content: str) -> dict:
    """Safely extract JSON object from content."""
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("No JSON found in content.")
        return json.loads(content[start:end])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e}")

def validate_output_schema(json_data):
    """Validate JSON data against the predefined schema."""
    try:
        validate(instance=json_data, schema=OUTPUT_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")

def extract_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None
    return text[start:end + 1]

# def normalize_output(raw_json):
#     """
#     Normalize the LLM output to match the required schema.
#     """
#     normalized_json = {
#         "overview": {
#             "total_attacks": len(raw_json.get("frequency_and_severity", [])),
#             "unique_attack_types": len(raw_json.get("attack_types", [])),
#             "most_frequent_attack": raw_json.get("attack_types", [{}])[0].get("type", "Unknown"),
#             "log_timeline": "To be extracted"
#         },
#         "attack_summary": [
#             {
#                 "attack_type": attack.get("type", "Unknown"),
#                 "count": attack.get("frequency", 0),
#                 "owasp_category": "Unknown OWASP Category",
#                 "owasp_description": "No description available",
#                 "severity": attack.get("severity", "Unknown")
#             }
#             for attack in raw_json.get("attack_types", [])
#         ],
#         "events": [
#             {
#                 "timestamp": record.get("time", "Unknown"),
#                 "source_ip": "Unknown",
#                 "destination_ip": "Unknown",
#                 "attack_type": record.get("description", "Unknown"),
#                 "vulnerability": "Not available",
#                 "affected_service": service.get("service", "Unknown"),
#                 "severity": record.get("severity", "Unknown"),
#                 "recommended_action": "No action specified"
#             }
#             for record in raw_json.get("frequency_and_severity", [])
#             for service in raw_json.get("affected_services_and_ports", [{}])
#         ]
#     }
#     return normalized_json

def normalize_events(events):
    normalized = []
    for e in events:
        normalized.append({
            "timestamp": e.get("timestamp", "unknown"),
            "source_ip": e.get("source_ip", "unknown"),
            "destination_ip": "SERVER_IP",
            "attack_type": e.get("attack_type", "SSH Brute Force"),
            "vulnerability": "Weak Authentication",
            "affected_service": "SSH",
            "severity": e.get("severity", "HIGH"),
            "recommended_action": e.get(
                "recommended_action",
                "Block IP, enable SSH key authentication, disable root login"
            )
        })
    return normalized