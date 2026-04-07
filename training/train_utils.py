# training/train_utils.py

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List


def append_training_log(logs: List[Dict], row: Dict) -> None:
    """
    Append one episode-level log row to the in-memory log list.
    """
    logs.append(row)


def save_logs_to_csv(logs: List[Dict], filepath: str) -> None:
    """
    Save logs as CSV.
    """
    if not logs:
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = list(logs[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)


def save_logs_to_json(logs: List[Dict], filepath: str) -> None:
    """
    Save logs as JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(logs, json_file, indent=2)