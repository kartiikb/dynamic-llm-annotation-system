import json
import time
from pathlib import Path

from loader import load_dataset
from prompt_builder import load_prompt_template, build_annotation_prompt
from annotator import LLMAnnotator
from validator import validate_annotation


# ============================================================
# PATHS
# ============================================================

CONFIG_PATH = "config/dictionary_validation_config.json"
DATASET_PATH = "data/culture_terms.json"
PROMPT_PATH = "prompts/annotation_prompt.txt"

# Existing annotations produced by all your previous runs/APIs
OUTPUT_PATH = "output/culture_terms_annotated.json"

# Temporary recovery output
RECOVERY_PATH = "output/culture_terms_recovery.json"

# Final ordered output
FINAL_OUTPUT_PATH = "output/culture_terms_annotated_final.json"


# ============================================================
# CONFIG
# ============================================================

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = file_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    temp_path.replace(file_path)


# ============================================================
# EXISTING ANNOTATIONS
# ============================================================

def get_annotation_map(results):
    """
    Convert saved annotations into:

        entry_id -> annotated entry

    This makes comparison independent of list order.
    """

    annotation_map = {}

    for item in results:

        if "entry_id" not in item:
            continue

        entry_id = str(item["entry_id"])

        annotation_map[entry_id] = item

    return annotation_map


# ============================================================
# FIND MISSING ENTRIES
# ============================================================

def find_missing_entries(dataset, annotation_map):
    """
    Compare the original dataset against existing annotations.

    Only entry_id is used for matching.
    """

    missing = []

    for entry in dataset:

        entry_id = str(entry["entry_id"])

        if entry_id not in annotation_map:
            missing.append(entry)

    return missing


# ============================================================
# MANUAL INPUT HELPERS
# ============================================================

def ask_non_empty(prompt):
    """
    Ask until the user provides a non-empty value.
    """

    while True:

        value = input(prompt).strip()

        if value:
            return value

        print("Please enter a value.")


def ask_choice(prompt, choices):
    """
    Ask until the user enters one of the allowed choices.
    """

    choices_lower = [str(x).lower() for x in choices]

    while True:

        value = input(prompt).strip().lower()

        if value in choices_lower:
            return value

        print(
            f"Invalid value. Choose one of: "
            f"{', '.join(choices)}"
        )


def ask_score(prompt):
    """
    Ask for an integer score from 1 to 5.
    """

    while True:

        value = input(prompt).strip()

        try:
            score = int(value)

            if score in [1, 2, 3, 4, 5]:
                return score

        except ValueError:
            pass

        print("Please enter a score from 1 to 5.")


def ask_optional(prompt):
    """
    Optional text field.
    Empty input is allowed.
    """

    return input(prompt).strip()


# ============================================================
# MANUAL ANNOTATION
# ============================================================

def manually_annotate(entry, config):
    """
    Collect one complete annotation manually from the user.

    The fields follow the README specification.
    """

    print("\n")
    print("=" * 70)
    print("MANUAL ANNOTATION REQUIRED")
    print("=" * 70)

    print(f"\nEntry ID:       {entry['entry_id']}")
    print(f"Kanda:          {entry.get('kanda', '')}")
    print(f"Source language: {entry.get('source_language', '')}")
    print(f"Source word:    {entry.get('source_word', '')}")
    print(f"Target language: {entry.get('target_language', '')}")
    print(f"Target word:    {entry.get('target_word', '')}")
    print(f"Culture aspect: {entry.get('culture_aspect', '')}")
    print(f"Gloss:          {entry.get('gloss', '')}")
    print(f"Sample:         {entry.get('sample', '')}")

    print("\n" + "-" * 70)
    print("Enter annotation")
    print("-" * 70)

    # --------------------------------------------------------
    # annotator_id
    # --------------------------------------------------------

    annotator_id = config["annotator_id"]

    print(f"\nannotator_id = {annotator_id}")

    # --------------------------------------------------------
    # is_culture_specific
    # --------------------------------------------------------

    is_culture_specific = ask_choice(
        "\nis_culture_specific [yes/no/borderline]: ",
        ["yes", "no", "borderline"]
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    if is_culture_specific in ["yes", "borderline"]:

        target_word_correct = ask_score(
            "\ntarget_word_correct [1-5]: "
        )

        gloss_accuracy = ask_score(
            "gloss_accuracy [1-5]: "
        )

    else:

        target_word_correct = ""
        gloss_accuracy = ""

    # --------------------------------------------------------
    # aspect_correct
    # --------------------------------------------------------

    aspect_correct = ask_choice(
        "\naspect_correct [yes/no]: ",
        ["yes", "no"]
    )

    # --------------------------------------------------------
    # suggested_aspect
    # --------------------------------------------------------

    if aspect_correct == "no":

        valid_aspects = config["output_fields"][
            "suggested_aspect"
        ]["allowed_values"]

        print(
            "\nValid aspects:"
        )

        for aspect in valid_aspects:
            print(f"  - {aspect}")

        suggested_aspect = ask_choice(
            "\nsuggested_aspect: ",
            valid_aspects
        )

    else:

        suggested_aspect = ""

    # --------------------------------------------------------
    # suggested_target
    # --------------------------------------------------------

    if target_word_correct in [1, 2]:

        suggested_target = ask_non_empty(
            "\nsuggested_target: "
        )

    else:

        suggested_target = ""

    # --------------------------------------------------------
    # comments
    # --------------------------------------------------------

    comments = ask_optional(
        "\ncomments [optional]: "
    )

    # --------------------------------------------------------
    # Build annotation
    # --------------------------------------------------------

    annotation = {
        "annotator_id": annotator_id,
        "is_culture_specific": is_culture_specific,
        "target_word_correct": target_word_correct,
        "gloss_accuracy": gloss_accuracy,
        "aspect_correct": aspect_correct,
        "suggested_aspect": suggested_aspect,
        "suggested_target": suggested_target,
        "comments": comments
    }

    return annotation


# ============================================================
# API ANNOTATION
# ============================================================

def annotate_with_api(
    entry,
    prompt_template,
    config,
    annotator
):
    """
    Try to annotate one entry using the API.

    Returns:

        annotation, None

    on success.

    Returns:

        None, error

    on failure.
    """

    prompt = build_annotation_prompt(
        entry,
        prompt_template,
        config["annotator_id"]
    )

    annotation = annotator.annotate(prompt)

    # --------------------------------------------------------
    # Normalize non-culture-specific entries
    # --------------------------------------------------------

    if annotation.get("is_culture_specific") == "no":

        annotation["target_word_correct"] = ""
        annotation["gloss_accuracy"] = ""

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    is_valid, errors = validate_annotation(
        annotation,
        config
    )

    if not is_valid:

        error_message = "\n".join(
            f"- {error}"
            for error in errors
        )

        raise ValueError(
            f"LLM annotation failed validation:\n"
            f"{error_message}"
        )

    return annotation, None


# ============================================================
# FINAL ORDERED OUTPUT
# ============================================================

def build_final_output(dataset, annotation_map):
    """
    Reconstruct the complete output in exactly the same
    order as the original dataset.

    Original dataset determines the order.
    Annotation map provides the annotation.
    """

    final_results = []

    missing_ids = []

    for entry in dataset:

        entry_id = str(entry["entry_id"])

        if entry_id not in annotation_map:

            missing_ids.append(entry_id)
            continue

        output_entry = entry.copy()

        # Add the annotation fields
        annotation = annotation_map[entry_id]

        for field in [
            "annotator_id",
            "is_culture_specific",
            "target_word_correct",
            "gloss_accuracy",
            "aspect_correct",
            "suggested_aspect",
            "suggested_target",
            "comments"
        ]:

            if field in annotation:
                output_entry[field] = annotation[field]

        final_results.append(output_entry)

    return final_results, missing_ids


# ============================================================
# MAIN RECOVERY PROCESS
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("MISSING ENTRY RECOVERY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------

    print("\nLoading configuration...")
    config = load_config()

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    print("Loading original dataset...")
    dataset = load_dataset(DATASET_PATH)

    # --------------------------------------------------------
    # Load prompt
    # --------------------------------------------------------

    print("Loading prompt...")
    prompt_template = load_prompt_template(
        PROMPT_PATH
    )

    # --------------------------------------------------------
    # Load existing annotations
    # --------------------------------------------------------

    print("Loading existing annotations...")

    existing_results = load_json_file(
        OUTPUT_PATH
    )

    if existing_results is None:
        existing_results = []

    if not isinstance(existing_results, list):
        raise ValueError(
            "Existing annotation output must be a JSON list."
        )

    print(
        f"Existing annotations: "
        f"{len(existing_results)}"
    )

    # --------------------------------------------------------
    # Build annotation map
    # --------------------------------------------------------

    annotation_map = get_annotation_map(
        existing_results
    )

    # --------------------------------------------------------
    # Find missing entries
    # --------------------------------------------------------

    missing_entries = find_missing_entries(
        dataset,
        annotation_map
    )

    print(
        f"Original entries:     {len(dataset)}"
    )

    print(
        f"Already annotated:    {len(annotation_map)}"
    )

    print(
        f"Missing entries:      {len(missing_entries)}"
    )

    # --------------------------------------------------------
    # Nothing missing
    # --------------------------------------------------------

    if not missing_entries:

        print(
            "\nNo missing entries."
        )

        final_results, missing_ids = build_final_output(
            dataset,
            annotation_map
        )

        if missing_ids:
            raise RuntimeError(
                f"Unexpected missing IDs: {missing_ids}"
            )

        save_json_file(
            final_results,
            FINAL_OUTPUT_PATH
        )

        print(
            f"\nFinal ordered file created:"
            f"\n{FINAL_OUTPUT_PATH}"
        )

        return

    # --------------------------------------------------------
    # Show missing IDs
    # --------------------------------------------------------

    print("\nMissing entry IDs:")

    print(
        ", ".join(
            str(entry["entry_id"])
            for entry in missing_entries
        )
    )

    # --------------------------------------------------------
    # Initialize API
    # --------------------------------------------------------

    print("\nInitializing API...")
    annotator = LLMAnnotator()

    # --------------------------------------------------------
    # Manual mode flag
    # --------------------------------------------------------

    manual_mode = False

    # --------------------------------------------------------
    # Recovery loop
    # --------------------------------------------------------

    for index, entry in enumerate(
        missing_entries,
        start=1
    ):

        entry_id = str(entry["entry_id"])

        print("\n")
        print("=" * 70)
        print(
            f"RECOVERY {index}/{len(missing_entries)}"
        )
        print(
            f"Entry ID: {entry_id}"
        )
        print("=" * 70)

        annotation = None

        # ====================================================
        # API MODE
        # ====================================================

        if not manual_mode:

            try:

                print(
                    "\nTrying API annotation..."
                )

                annotation, _ = annotate_with_api(
                    entry,
                    prompt_template,
                    config,
                    annotator
                )

                print(
                    f"✓ API annotation successful "
                    f"for {entry_id}"
                )

            except Exception as e:

                print(
                    "\n✗ API annotation failed."
                )

                print(
                    f"Reason: {e}"
                )

                print(
                    "\nSwitching to MANUAL MODE."
                )

                print(
                    "All remaining missing entries "
                    "will now be entered manually."
                )

                manual_mode = True

        # ====================================================
        # MANUAL MODE
        # ====================================================

        if manual_mode:

            while True:

                annotation = manually_annotate(
                    entry,
                    config
                )

                # --------------------------------------------
                # Validate manual annotation
                # --------------------------------------------

                is_valid, errors = validate_annotation(
                    annotation,
                    config
                )

                if is_valid:

                    print(
                        "\n✓ Manual annotation is valid."
                    )

                    break

                print(
                    "\n✗ Manual annotation is invalid:"
                )

                for error in errors:
                    print(
                        f"  - {error}"
                    )

                print(
                    "\nPlease enter this entry again."
                )

        # ====================================================
        # SAVE RECOVERED ENTRY
        # ====================================================

        output_entry = entry.copy()

        output_entry.update(
            annotation
        )

        # Update map
        annotation_map[entry_id] = output_entry

        # Save recovery progress immediately
        save_json_file(
            list(annotation_map.values()),
            RECOVERY_PATH
        )

        print(
            f"\n✓ Saved recovered entry {entry_id}"
        )

        print(
            f"Progress: "
            f"{index}/{len(missing_entries)}"
        )

        # Small delay only after API calls
        if not manual_mode:
            time.sleep(1)

    # ========================================================
    # BUILD FINAL ORDERED OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BUILDING FINAL ORDERED OUTPUT")
    print("=" * 70)

    final_results, missing_ids = build_final_output(
        dataset,
        annotation_map
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if missing_ids:

        print(
            "\nERROR: Some entries are still missing:"
        )

        print(
            ", ".join(missing_ids)
        )

        raise RuntimeError(
            "Final output was NOT created because "
            "some original entries are still missing."
        )

    # --------------------------------------------------------
    # Verify count
    # --------------------------------------------------------

    if len(final_results) != len(dataset):

        raise RuntimeError(
            f"Final output contains "
            f"{len(final_results)} entries, "
            f"but original dataset contains "
            f"{len(dataset)} entries."
        )

    # --------------------------------------------------------
    # Save final output
    # --------------------------------------------------------

    save_json_file(
        final_results,
        FINAL_OUTPUT_PATH
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)

    print(
        f"Original entries:      {len(dataset)}"
    )

    print(
        f"Final entries:         {len(final_results)}"
    )

    print(
        f"Missing after recovery: {len(missing_ids)}"
    )

    print(
        f"\nFinal file:"
    )

    print(
        f"{FINAL_OUTPUT_PATH}"
    )

    print(
        "\n✓ All entries are present."
    )

    print(
        "✓ Original order has been restored."
    )


if __name__ == "__main__":
    main()