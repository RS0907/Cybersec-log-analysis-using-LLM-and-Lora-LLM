import json
from pathlib import Path

def load_text(file_path: Path) -> str:
    """Load text content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def save_json(file_path: Path, data: dict):
    """Save dictionary as a JSON file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)