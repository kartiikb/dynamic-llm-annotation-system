import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from item_check_validator import validate_annotation, is_complete_valid_row

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "item_check_config.json"
load_dotenv(ROOT / ".env")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_safely(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(path)


def parse_json_object(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("The model response was not a JSON object.")
    return result


def build_prompt(template, row, annotator_id):
    row_json = json.dumps(
        {key: row.get(key, "") for key in [
            "check_id", "source_text", "translation", "item_term",
            "reference_rendering", "category", "gloss"
        ]},
        ensure_ascii=False,
        indent=2
    )
    return (
        f"{template}\n\n"
        f"Annotator ID: {annotator_id}\n"
        f"Annotate this row:\n{row_json}\n"
        "Return only the required JSON object."
    )


def api_annotation(client, model, prompt, annotator_id):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content or ""
    result = parse_json_object(content)
    ok, errors = validate_annotation(result, annotator_id)
    if not ok:
        raise ValueError("Annotation validation failed: " + "; ".join(errors))
    return result


def manual_annotation(row, annotator_id):
    print("\n" + "=" * 78)
    print(f"MANUAL ANNOTATION — {row.get('check_id')}")
    print("=" * 78)
    for field in [
        "source_text", "translation", "item_term",
        "reference_rendering", "category", "gloss"
    ]:
        print(f"\n{field.upper()}:\n{row.get(field, '')}")

    print("\nSeverity: none / minor / major / critical / not_applicable / uncertain")
    print("Strategy: RET / EQV / EXP / GEN / NEU / SUB / OMI / DIS / None")
    while True:
        severity = input("severity: ").strip()
        strategy = input("strategy: ").strip()
        comments = input("comments (required for major/critical/uncertain): ").strip()
        result = {
            "annotator_id": annotator_id,
            "severity": severity,
            "strategy": strategy,
            "comments": comments
        }
        ok, errors = validate_annotation(result, annotator_id)
        if ok:
            return result
        print("\nPlease correct these issues:")
        for error in errors:
            print(f" - {error}")
        print("Enter all four fields again.\n")


def main():
    config = read_json(CONFIG_PATH)
    annotator_id = config["annotator_id"]
    model = config["model"]
    input_path = ROOT / config["input_file"]
    output_path = ROOT / config["output_file"]
    prompt_path = ROOT / config["prompt_file"]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Place CIFE_item_check_telugu.json in the project's data/ folder."
        )

    original = read_json(input_path)
    if not isinstance(original, list):
        raise ValueError("Input JSON must be a list of rows.")
    if len(original) != 100:
        print(f"WARNING: README says 100 rows, but input has {len(original)} rows.")

    ids = [row.get("check_id") for row in original]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Input has missing or duplicate check_id values.")

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Keep the original dataset as the base. Only annotation fields are copied
    # from a prior output, so source fields cannot be changed by the model.
    output_by_id = {}
    if output_path.exists():
        previous = read_json(output_path)
        if isinstance(previous, list):
            for row in previous:
                if isinstance(row, dict) and row.get("check_id") in set(ids):
                    output_by_id[row["check_id"]] = row

    merged = []
    for source_row in original:
        row = dict(source_row)
        old = output_by_id.get(source_row["check_id"], {})
        for field in ["annotator_id", "severity", "strategy", "comments"]:
            if field in old:
                row[field] = old[field]
        merged.append(row)

    completed = {
        row["check_id"] for row in merged
        if is_complete_valid_row(row, annotator_id)
    }
    print(f"Input rows: {len(original)}")
    print(f"Already completed valid rows: {len(completed)}")
    print(f"Remaining rows: {len(original) - len(completed)}")
    print(f"Output: {output_path}")

    client = None
    api_enabled = True
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nGROQ_API_KEY was not found. Starting terminal manual mode.")
        api_enabled = False
    else:
        client = Groq(api_key=api_key)

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    for index, row in enumerate(merged, start=1):
        if row["check_id"] in completed:
            continue

        print(f"\n[{index}/{len(merged)}] {row['check_id']}")
        annotation = None

        if api_enabled:
            prompt = build_prompt(template, row, annotator_id)
            for attempt in range(1, int(config.get("max_api_attempts_per_row", 2)) + 1):
                try:
                    annotation = api_annotation(client, model, prompt, annotator_id)
                    break
                except Exception as exc:
                    print(f"API attempt {attempt} failed: {exc}")
            if annotation is None:
                print("\nAPI failed for this row.")
                choice = input(
                    "Switch to manual terminal mode for this and all remaining rows? [Y/n]: "
                ).strip().lower()
                if choice in {"", "y", "yes"}:
                    api_enabled = False
                else:
                    print("Stopping safely. Progress has been saved.")
                    write_json_safely(output_path, merged)
                    return

        if not api_enabled:
            annotation = manual_annotation(row, annotator_id)

        # Apply only the four allowed annotation fields.
        for field in ["annotator_id", "severity", "strategy", "comments"]:
            row[field] = annotation[field]

        ok, errors = validate_annotation(
            {field: row[field] for field in ["annotator_id", "severity", "strategy", "comments"]},
            annotator_id
        )
        if not ok:
            print("Annotation did not pass validation:")
            for error in errors:
                print(f" - {error}")
            print("Stopping safely. This row was not marked complete.")
            write_json_safely(output_path, merged)
            return

        write_json_safely(output_path, merged)
        completed.add(row["check_id"])
        print(f"Saved {len(completed)}/{len(merged)} valid annotations.")

    print("\nANNOTATION WORKFLOW FINISHED")
    print(f"Output saved to: {output_path}")
    print("Next run: python src/item_check_final_validation.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Any previously saved annotations remain in the output file.")
        sys.exit(1)
