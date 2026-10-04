import json
from pathlib import Path


def load_dataset(file_path):
    """
    Load the dictionary validation JSON dataset.

    Returns:
        list: List of dictionary entries.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Expected the JSON dataset to contain a list of entries."
        )

    print(f"Loaded {len(data)} entries.")

    return data


def validate_input_fields(data, required_fields):
    """
    Check that every entry contains the required input fields.
    """

    if not data:
        raise ValueError("Dataset is empty.")

    missing_fields = []

    for index, entry in enumerate(data):
        for field in required_fields:
            if field not in entry:
                missing_fields.append(
                    f"Entry {index}: missing '{field}'"
                )

    if missing_fields:
        raise ValueError(
            "Missing required fields:\n" +
            "\n".join(missing_fields)
        )

    print("Input field validation passed.")


def check_entry_ids(data):
    """
    Check that entry_id exists and is unique.
    """

    entry_ids = [entry["entry_id"] for entry in data]

    if any(entry_id is None for entry_id in entry_ids):
        raise ValueError("At least one entry has a missing entry_id.")

    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("Duplicate entry_id values found.")

    print("Entry ID validation passed.")


if __name__ == "__main__":

    dataset_path = "data/culture_terms.json"

    data = load_dataset(dataset_path)

    required_fields = [
        "entry_id",
        "kanda",
        "source_language",
        "source_word",
        "target_language",
        "target_word",
        "culture_aspect",
        "gloss",
        "sample"
    ]

    validate_input_fields(data, required_fields)
    check_entry_ids(data)

    print("\nFirst entry:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))