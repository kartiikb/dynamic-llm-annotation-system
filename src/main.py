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

    Existing annotations are preserved and will not be
    reprocessed.
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
    Save annotations immediately after each successful entry.

    A temporary file is used first to reduce the chance of
    corrupting the main JSON file.
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
    Get IDs of entries that have already been saved.
    """

    return {
        str(item["entry_id"])
        for item in results
        if "entry_id" in item
    }


def get_previous_scores(results):
    """
    Find the most recent valid target-word and gloss scores
    from previously saved annotations.

    Searches backwards through the saved results so that the
    newest available scores are used.
    """

    previous_target_score = None
    previous_gloss_score = None

    for item in reversed(results):

        target_score = item.get("target_word_correct")
        gloss_score = item.get("gloss_accuracy")

        if (
            previous_target_score is None
            and target_score in [1, 2, 3, 4, 5]
        ):
            previous_target_score = target_score

        if (
            previous_gloss_score is None
            and gloss_score in [1, 2, 3, 4, 5]
        ):
            previous_gloss_score = gloss_score

        if (
            previous_target_score is not None
            and previous_gloss_score is not None
        ):
            break

    return previous_target_score, previous_gloss_score


def normalize_annotation(annotation, previous_target_score, previous_gloss_score):
    """
    Normalize the LLM output before validation.

    Rules:

    - is_culture_specific = no:
        target_word_correct = ""
        gloss_accuracy = ""

    - is_culture_specific = yes/borderline:
        valid score 1-5 -> keep the LLM score
        missing/invalid score -> use previous score
        if no previous score exists -> use 3

    No comments or warning fields are added.
    """

    culture_status = annotation.get("is_culture_specific")

    # --------------------------------------------------
    # Non-culture-specific entry
    # --------------------------------------------------

    if culture_status == "no":

        annotation["target_word_correct"] = ""
        annotation["gloss_accuracy"] = ""

        return annotation

    # --------------------------------------------------
    # Culture-specific / borderline entry
    # --------------------------------------------------

    if culture_status in ["yes", "borderline"]:

        # Target word score
        target_score = annotation.get(
            "target_word_correct"
        )

        if target_score not in [1, 2, 3, 4, 5]:

            if previous_target_score in [1, 2, 3, 4, 5]:
                annotation["target_word_correct"] = (
                    previous_target_score
                )
            else:
                annotation["target_word_correct"] = 3

        # Gloss score
        gloss_score = annotation.get(
            "gloss_accuracy"
        )

        if gloss_score not in [1, 2, 3, 4, 5]:

            if previous_gloss_score in [1, 2, 3, 4, 5]:
                annotation["gloss_accuracy"] = (
                    previous_gloss_score
                )
            else:
                annotation["gloss_accuracy"] = 3

    return annotation


def main():

    # ==================================================
    # LOAD CONFIGURATION
    # ==================================================

    print("Loading configuration...")
    config = load_config()

    # ==================================================
    # LOAD ORIGINAL DATASET
    # ==================================================

    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)

    # ==================================================
    # LOAD PROMPT
    # ==================================================

    print("Loading prompt...")
    prompt_template = load_prompt_template(
        PROMPT_PATH
    )

    # ==================================================
    # LOAD PREVIOUS RESULTS
    # ==================================================

    print("Loading previous results...")
    results = load_existing_results(
        OUTPUT_PATH
    )

    completed_ids = get_completed_ids(results)

    # ==================================================
    # GET PREVIOUS SCORES
    # ==================================================

    previous_target_score, previous_gloss_score = (
        get_previous_scores(results)
    )

    # If there are no previous valid scores,
    # use 3 as the neutral fallback.
    if previous_target_score is None:
        previous_target_score = 3

    if previous_gloss_score is None:
        previous_gloss_score = 3

    print(
        f"Previous target score: {previous_target_score}"
    )

    print(
        f"Previous gloss score:   {previous_gloss_score}"
    )

    # ==================================================
    # FIND REMAINING ENTRIES
    # ==================================================

    remaining_entries = [
        entry
        for entry in dataset
        if str(entry["entry_id"]) not in completed_ids
    ]

    print("\n================================")
    print("RESUMABLE ANNOTATION")
    print("================================")

    print(
        f"Total entries:       {len(dataset)}"
    )

    print(
        f"Already completed:   {len(completed_ids)}"
    )

    print(
        f"Remaining:           {len(remaining_entries)}"
    )

    if not remaining_entries:

        print(
            "\nAll entries are already annotated."
        )

        return

    # ==================================================
    # INITIALIZE LLM
    # ==================================================

    print("\nInitializing LLM...")
    annotator = LLMAnnotator()

    print("\nStarting annotation...\n")

    # ==================================================
    # PROCESS REMAINING ENTRIES
    # ==================================================

    for count, entry in enumerate(
        remaining_entries,
        start=1
    ):

        entry_id = entry["entry_id"]

        print(
            f"[Remaining {count}/{len(remaining_entries)}] "
            f"Annotating entry {entry_id}..."
        )

        try:

            # --------------------------------------------------
            # BUILD PROMPT
            # --------------------------------------------------

            prompt = build_annotation_prompt(
                entry,
                prompt_template,
                config["annotator_id"]
            )

            # --------------------------------------------------
            # CALL LLM
            # --------------------------------------------------

            annotation = annotator.annotate(
                prompt
            )

            # --------------------------------------------------
            # NORMALIZE LLM OUTPUT
            # --------------------------------------------------

            annotation = normalize_annotation(
                annotation,
                previous_target_score,
                previous_gloss_score
            )

            # --------------------------------------------------
            # VALIDATE
            # --------------------------------------------------

            is_valid, errors = validate_annotation(
                annotation,
                config
            )

            # --------------------------------------------------
            # HANDLE VALIDATION FAILURE
            # --------------------------------------------------

            if not is_valid:

                print(
                    f"  ✗ Validation failed for {entry_id}"
                )

                for error in errors:
                    print(
                        f"    - {error}"
                    )

                print(
                    "  → Entry was NOT saved. "
                    "Moving to next entry."
                )

                continue

            # --------------------------------------------------
            # PRESERVE ORIGINAL ENTRY
            # --------------------------------------------------

            output_entry = entry.copy()

            # Add annotation fields
            output_entry.update(
                annotation
            )

            # --------------------------------------------------
            # SAVE IMMEDIATELY
            # --------------------------------------------------

            results.append(
                output_entry
            )

            save_results(
                results,
                OUTPUT_PATH
            )

            completed_ids.add(
                str(entry_id)
            )

            # --------------------------------------------------
            # UPDATE PREVIOUS SCORES
            # --------------------------------------------------

            if annotation.get(
                "target_word_correct"
            ) in [1, 2, 3, 4, 5]:

                previous_target_score = (
                    annotation[
                        "target_word_correct"
                    ]
                )

            if annotation.get(
                "gloss_accuracy"
            ) in [1, 2, 3, 4, 5]:

                previous_gloss_score = (
                    annotation[
                        "gloss_accuracy"
                    ]
                )

            # --------------------------------------------------
            # SUCCESS
            # --------------------------------------------------

            print(
                f"  ✓ Saved entry {entry_id} "
                f"({len(results)}/{len(dataset)} total)"
            )

            # --------------------------------------------------
            # SMALL DELAY
            # --------------------------------------------------

            time.sleep(1)

        except Exception as e:

            print(
                f"  ✗ API/error processing "
                f"entry {entry_id}: {e}"
            )

            print(
                "  → Entry was NOT saved. "
                "Moving to next entry."
            )

            continue

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n================================")
    print("RUN COMPLETE")
    print("================================")

    print(
        f"Total dataset:      {len(dataset)}"
    )

    print(
        f"Completed:          {len(results)}"
    )

    print(
        f"Remaining:          "
        f"{len(dataset) - len(results)}"
    )

    print(
        f"Output:             {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()