# utils/logger.py

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List


def ensure_parent_dir(filepath: str) -> None:
    """
    Create the parent directory for a file path if it does not exist.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def save_json(data: Any, filepath: str) -> None:
    """
    Save data as JSON.
    """
    ensure_parent_dir(filepath)

    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def save_csv_rows(rows: List[Dict], filepath: str) -> None:
    """
    Save a list of dictionaries as CSV.

    This version supports rows that may not all have the exact same keys.
    It builds the union of all keys across all rows, preserves first-seen
    column order, and fills missing values with empty strings.
    """
    if not rows:
        return

    ensure_parent_dir(filepath)

    # Collect the union of all field names across every row
    # while preserving the order in which keys first appear.
    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(filepath, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            # Fill in any missing columns with blank values
            complete_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(complete_row)


def append_log_row(log_rows: List[Dict], row: Dict) -> None:
    """
    Append one row to an in-memory log list.
    """
    log_rows.append(row)