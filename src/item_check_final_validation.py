import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "item_check_config.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    config = load_json(CONFIG_PATH)
    original_path = ROOT / config["input_file"]
    final_path = ROOT / config["output_file"]
    annotator_id = config["annotator_id"]
    annotation_fields = ["annotator_id", "severity", "strategy", "comments"]
    immutable_fields = [
        "check_id", "source_text", "translation", "item_term",
        "reference_rendering", "category", "gloss"
    ]
    severities = set(config["severity_values"])
    strategies = set(config["strategy_values"])
    errors = []

    original = load_json(original_path)
    if not final_path.exists():
        print(f"FAIL: output file not found: {final_path}")
        sys.exit(1)
    final = load_json(final_path)

    print("=" * 70)
    print("FINAL ITEM-CHECK VALIDATION")
    print("=" * 70)
    print(f"Original rows: {len(original)}")
    print(f"Final rows:    {len(final)}")

    if len(original) != 100:
        errors.append(f"Expected 100 original rows, found {len(original)}.")
    if len(final) != len(original):
        errors.append("Final row count does not match original.")

    original_ids = [r.get("check_id") for r in original]
    final_ids = [r.get("check_id") for r in final]
    if len(original_ids) != len(set(original_ids)):
        errors.append("Original file has duplicate check_id values.")
    if len(final_ids) != len(set(final_ids)):
        errors.append("Final file has duplicate check_id values.")
    if original_ids != final_ids:
        errors.append("Final check_id sequence differs from original.")

    for i, (src, out) in enumerate(zip(original, final), start=1):
        for field in immutable_fields:
            if src.get(field) != out.get(field):
                errors.append(f"Row {i} ({src.get('check_id')}): changed source field {field!r}.")
        for field in annotation_fields:
            if field not in out:
                errors.append(f"Row {i} ({src.get('check_id')}): missing {field!r}.")

        if out.get("annotator_id") != annotator_id:
            errors.append(f"Row {i} ({src.get('check_id')}): incorrect annotator_id.")

        severity = out.get("severity")
        strategy = out.get("strategy")
        comments = out.get("comments")

        if severity not in severities:
            errors.append(f"Row {i} ({src.get('check_id')}): invalid severity {severity!r}.")
        if strategy not in strategies:
            errors.append(f"Row {i} ({src.get('check_id')}): invalid strategy {strategy!r}.")
        if not isinstance(comments, str):
            errors.append(f"Row {i} ({src.get('check_id')}): comments must be a string.")
        elif severity in {"major", "critical", "uncertain"} and not comments.strip():
            errors.append(f"Row {i} ({src.get('check_id')}): comment required for {severity}.")

        if severity in {"not_applicable", "uncertain"}:
            if strategy != "None":
                errors.append(f"Row {i} ({src.get('check_id')}): strategy must be 'None' for {severity}.")
        elif strategy == "None":
            errors.append(f"Row {i} ({src.get('check_id')}): 'None' strategy used with {severity}.")

    if errors:
        print(f"\nFAIL: {len(errors)} issue(s) found.")
        for error in errors[:100]:
            print(" - " + error)
        if len(errors) > 100:
            print(f" ... and {len(errors) - 100} more.")
        sys.exit(1)

    print("\nPASS")
    print("✓ 100 rows")
    print("✓ Same IDs and original order")
    print("✓ No duplicate IDs")
    print("✓ Original fields unchanged")
    print("✓ All four annotation fields valid")
    print("✓ Major/critical/uncertain comments present")
    print("✓ Severity/strategy None rule satisfied")
    print("✓ Valid JSON")
    print(f"\nFinal file: {final_path}")

if __name__ == "__main__":
    main()
