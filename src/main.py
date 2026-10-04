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


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved {len(results)} entries to {output_path}")


def main():

    print("Loading configuration...")
    config = load_config()

    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)

    print("Loading prompt...")
    prompt_template = load_prompt_template(PROMPT_PATH)

    print("Initializing LLM...")
    annotator = LLMAnnotator()

    results = []

    total = len(dataset)

    print(f"\nStarting annotation of {total} entries...\n")

    for index, entry in enumerate(dataset):

        entry_id = entry["entry_id"]

        print(
            f"[{index + 1}/{total}] "
            f"Annotating entry {entry_id}..."
        )

        try:

            # Build prompt
            prompt = build_annotation_prompt(
                entry,
                prompt_template,
                config["annotator_id"]
            )

            # Call LLM
            annotation = annotator.annotate(prompt)

            # Validate
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

                # Keep original entry but mark annotation failure
                annotation = {
                    "annotator_id": config["annotator_id"],
                    "is_culture_specific": "",
                    "target_word_correct": "",
                    "gloss_accuracy": "",
                    "aspect_correct": "",
                    "suggested_aspect": "",
                    "suggested_target": "",
                    "comments": (
                        "ANNOTATION_FAILED: "
                        + " | ".join(errors)
                    )
                }

            else:

                print(
                    f"  ✓ Valid annotation for {entry_id}"
                )

            # Copy original entry
            output_entry = entry.copy()

            # Add annotation fields
            output_entry.update(annotation)

            results.append(output_entry)

            # Save after every entry
            save_results(results, OUTPUT_PATH)

            # Small delay
            time.sleep(1)

        except Exception as e:

            print(
                f"  ✗ ERROR processing {entry_id}: {e}"
            )

            output_entry = entry.copy()

            output_entry.update({
                "annotator_id": config["annotator_id"],
                "is_culture_specific": "",
                "target_word_correct": "",
                "gloss_accuracy": "",
                "aspect_correct": "",
                "suggested_aspect": "",
                "suggested_target": "",
                "comments": f"ANNOTATION_ERROR: {str(e)}"
            })

            results.append(output_entry)

            save_results(results, OUTPUT_PATH)

    print("\n================================")
    print("ANNOTATION COMPLETE")
    print("================================")
    print(f"Total entries: {total}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()