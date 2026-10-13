import json
import time
from pathlib import Path

from loader import load_dataset
from prompt_builder import load_prompt_template, build_annotation_prompt
from annotator import LLMAnnotator
from validator import validate_annotation


CONFIG_PATH = "config/dictionary_validation_config.json"
DATASET_PATH = "data/culture_terms.json"
PROMPT_PATH = "prompts/annotation_prompt.txt"
OUTPUT_PATH = "output/culture_terms_annotated.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results(output_path):
    """
    Load previously completed annotations.

    If no output file exists, start with an empty list.
    """

    output_path = Path(output_path)

    if not output_path.exists():
        print("No previous annotation file found.")
        return []

    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError(
            "Existing annotation file must contain a JSON list."
        )

    print(
        f"Loaded {len(results)} previously completed annotations."
    )

    return results


def save_results(results, output_path):
    """
    Save all completed annotations.

    Uses a temporary file first so that the main JSON file
    is less likely to be left incomplete if something goes wrong.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    temp_path.replace(output_path)


def get_completed_ids(results):
    """
    Get entry IDs that have already been saved.
    """

    return {
        str(item["entry_id"])
        for item in results
        if "entry_id" in item
    }


def main():

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    print("Loading configuration...")
    config = load_config()

    # --------------------------------------------------
    # Load original dataset
    # --------------------------------------------------

    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)

    # --------------------------------------------------
    # Load annotation prompt
    # --------------------------------------------------

    print("Loading prompt...")
    prompt_template = load_prompt_template(PROMPT_PATH)

    # --------------------------------------------------
    # Load previous annotations
    # --------------------------------------------------

    print("Loading previous results...")
    results = load_existing_results(OUTPUT_PATH)

    completed_ids = get_completed_ids(results)

    # --------------------------------------------------
    # Find entries that still need annotation
    # --------------------------------------------------

    remaining_entries = [
        entry
        for entry in dataset
        if str(entry["entry_id"]) not in completed_ids
    ]

    print("\n================================")
    print("RESUMABLE ANNOTATION")
    print("================================")

    print(f"Total entries:       {len(dataset)}")
    print(f"Already completed:   {len(completed_ids)}")
    print(f"Remaining:           {len(remaining_entries)}")

    if not remaining_entries:
        print("\nAll entries are already annotated.")
        return

    # --------------------------------------------------
    # Initialize LLM
    # --------------------------------------------------

    print("\nInitializing LLM...")
    annotator = LLMAnnotator()

    print("\nStarting annotation...\n")

    # --------------------------------------------------
    # Process remaining entries
    # --------------------------------------------------

    for count, entry in enumerate(remaining_entries, start=1):

        entry_id = entry["entry_id"]

        print(
            f"[Remaining {count}/{len(remaining_entries)}] "
            f"Annotating entry {entry_id}..."
        )

        try:

            # --------------------------------------------------
            # Build prompt
            # --------------------------------------------------

            prompt = build_annotation_prompt(
                entry,
                prompt_template,
                config["annotator_id"]
            )

            # --------------------------------------------------
            # Call LLM
            # --------------------------------------------------

            annotation = annotator.annotate(prompt)

            # --------------------------------------------------
            # Normalize non-culture-specific entries
            # --------------------------------------------------
            #
            # README allows these quality fields to be blank
            # when is_culture_specific = "no".
            #
            # If the LLM unnecessarily gives scores, we replace
            # them with blank values before validation.
            # --------------------------------------------------

            if annotation.get("is_culture_specific") == "no":

                annotation["target_word_correct"] = ""
                annotation["gloss_accuracy"] = ""

                print(
                    "  → Non-culture-specific entry: "
                    "target_word_correct and gloss_accuracy "
                    "set to blank."
                )

            # --------------------------------------------------
            # Validate annotation
            # --------------------------------------------------

            is_valid, errors = validate_annotation(
                annotation,
                config
            )

            if not is_valid:

                print(
                    f"  ✗ Validation failed for {entry_id}"
                )

                for error in errors:
                    print(f"    - {error}")

                print(
                    "  → Entry was NOT saved. "
                    "It will be retried on the next run."
                )

                continue

            # --------------------------------------------------
            # Preserve original entry
            # --------------------------------------------------

            output_entry = entry.copy()

            # Add annotation fields
            output_entry.update(annotation)

            # --------------------------------------------------
            # Save immediately
            # --------------------------------------------------

            results.append(output_entry)

            save_results(
                results,
                OUTPUT_PATH
            )

            completed_ids.add(
                str(entry_id)
            )

            print(
                f"  ✓ Saved entry {entry_id} "
                f"({len(results)}/{len(dataset)} total)"
            )

            # --------------------------------------------------
            # Small delay between API calls
            # --------------------------------------------------

            time.sleep(1)

        except Exception as e:

            print(
                f"  ✗ API/error processing entry {entry_id}: {e}"
            )

            print(
                "  → Entry was NOT saved. "
                "It will be retried on the next run."
            )

            # Continue with the next entry
            continue

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n================================")
    print("RUN COMPLETE")
    print("================================")

    print(f"Total dataset:      {len(dataset)}")
    print(f"Completed:          {len(results)}")
    print(
        f"Remaining:          "
        f"{len(dataset) - len(results)}"
    )
    print(f"Output:             {OUTPUT_PATH}")


if __name__ == "__main__":
    main()