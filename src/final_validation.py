import json
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

ORIGINAL_PATH = Path("data/culture_terms.json")
FINAL_PATH = Path("output/culture_terms_annotated_final.json")


# ============================================================
# EXPECTED VALUES FROM README
# ============================================================

EXPECTED_COUNT = 808
EXPECTED_ANNOTATOR_ID = "HUMAN_ANNOTATOR_01"

ANNOTATION_FIELDS = [
    "annotator_id",
    "is_culture_specific",
    "target_word_correct",
    "gloss_accuracy",
    "aspect_correct",
    "suggested_aspect",
    "suggested_target",
    "comments",
]

SOURCE_FIELDS = [
    "entry_id",
    "kanda",
    "source_language",
    "source_word",
    "target_language",
    "target_word",
    "culture_aspect",
    "gloss",
    "sample",
]

VALID_CULTURE_VALUES = {
    "yes",
    "no",
    "borderline",
}

VALID_ASPECT_VALUES = {
    "Ecological Culture",
    "Material Culture",
    "Social Culture",
    "Religious Culture",
    "Linguistic Culture",
}

VALID_ASPECT_CORRECT = {
    "yes",
    "no",
}


# ============================================================
# HELPERS
# ============================================================

def is_blank(value):
    return value is None or value == ""


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MAIN VALIDATION
# ============================================================

def validate():

    print("=" * 70)
    print("FINAL DICTIONARY VALIDATION")
    print("=" * 70)

    errors = []
    warnings = []

    # --------------------------------------------------------
    # 1. LOAD FILES
    # --------------------------------------------------------

    print("\nLoading files...")

    try:
        original = load_json(ORIGINAL_PATH)
        final = load_json(FINAL_PATH)
    except Exception as e:
        print(f"\n✗ FAILED TO LOAD FILE: {e}")
        return

    print("✓ Both JSON files loaded successfully.")

    # --------------------------------------------------------
    # 2. JSON STRUCTURE
    # --------------------------------------------------------

    if not isinstance(original, list):
        errors.append("Original dataset is not a JSON list.")

    if not isinstance(final, list):
        errors.append("Final dataset is not a JSON list.")

    if errors:
        print("\n✗ FAILED")
        for error in errors:
            print(f"  - {error}")
        return

    # --------------------------------------------------------
    # 3. COUNT
    # --------------------------------------------------------

    print("\n[1] ENTRY COUNT")

    print(f"  Original: {len(original)}")
    print(f"  Final:    {len(final)}")

    if len(original) != EXPECTED_COUNT:
        errors.append(
            f"Original dataset should contain {EXPECTED_COUNT} entries, "
            f"but contains {len(original)}."
        )

    if len(final) != EXPECTED_COUNT:
        errors.append(
            f"Final dataset should contain {EXPECTED_COUNT} entries, "
            f"but contains {len(final)}."
        )

    if len(original) == len(final) == EXPECTED_COUNT:
        print("  ✓ Count = 808")

    # --------------------------------------------------------
    # 4. ENTRY IDS
    # --------------------------------------------------------

    print("\n[2] ENTRY IDs")

    original_ids = [entry.get("entry_id") for entry in original]
    final_ids = [entry.get("entry_id") for entry in final]

    # Missing IDs
    if any(entry_id is None for entry_id in final_ids):
        errors.append("At least one final entry is missing entry_id.")

    # Duplicate IDs
    if len(final_ids) != len(set(final_ids)):
        errors.append("Duplicate entry_id values found in final output.")
        print("  ✗ Duplicate IDs found.")
    else:
        print("  ✓ No duplicate IDs.")

    # Same IDs
    if set(original_ids) != set(final_ids):
        missing = set(original_ids) - set(final_ids)
        extra = set(final_ids) - set(original_ids)

        if missing:
            errors.append(f"Missing entry IDs: {sorted(missing)}")

        if extra:
            errors.append(f"Unexpected entry IDs: {sorted(extra)}")

        print("  ✗ Entry ID sets do not match.")
    else:
        print("  ✓ Same entry IDs as original.")

    # --------------------------------------------------------
    # 5. ORDER
    # --------------------------------------------------------

    print("\n[3] ORIGINAL ORDER")

    if original_ids == final_ids:
        print("  ✓ Entry order is exactly the same as original.")
    else:
        errors.append("Entry order differs from original dataset.")

        # Show first mismatch
        for i, (orig_id, final_id) in enumerate(
            zip(original_ids, final_ids)
        ):
            if orig_id != final_id:
                print(
                    f"  ✗ First mismatch at position {i + 1}: "
                    f"original={orig_id}, final={final_id}"
                )
                break

    # --------------------------------------------------------
    # 6. ORIGINAL SOURCE DATA UNCHANGED
    # --------------------------------------------------------

    print("\n[4] ORIGINAL SOURCE DATA")

    source_changed = 0

    for i, (orig, fin) in enumerate(zip(original, final), start=1):

        for field in SOURCE_FIELDS:

            if orig.get(field) != fin.get(field):

                source_changed += 1

                errors.append(
                    f"Source field changed at row {i}, "
                    f"entry_id={orig.get('entry_id')}: "
                    f"'{field}'"
                )

    if source_changed == 0:
        print("  ✓ All original/source fields are unchanged.")
    else:
        print(f"  ✗ {source_changed} source-field changes detected.")

    # --------------------------------------------------------
    # 7. CHECK ANNOTATION FIELDS
    # --------------------------------------------------------

    print("\n[5] ANNOTATION FIELDS")

    missing_field_count = 0

    for i, entry in enumerate(final, start=1):

        for field in ANNOTATION_FIELDS:

            if field not in entry:

                missing_field_count += 1

                errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"missing annotation field '{field}'"
                )

    if missing_field_count == 0:
        print("  ✓ All 8 annotation fields exist in every row.")
    else:
        print(
            f"  ✗ {missing_field_count} missing annotation fields found."
        )

    # --------------------------------------------------------
    # 8. ANNOTATOR ID
    # --------------------------------------------------------

    print("\n[6] ANNOTATOR ID")

    bad_annotator_ids = []

    for i, entry in enumerate(final, start=1):

        value = entry.get("annotator_id")

        if value != EXPECTED_ANNOTATOR_ID:
            bad_annotator_ids.append(
                (i, entry.get("entry_id"), value)
            )

    if not bad_annotator_ids:
        print(
            f"  ✓ Every row has annotator_id = "
            f"{EXPECTED_ANNOTATOR_ID}"
        )
    else:
        errors.append(
            f"{len(bad_annotator_ids)} rows have an invalid annotator_id."
        )
        print(
            f"  ✗ {len(bad_annotator_ids)} invalid annotator IDs."
        )

    # --------------------------------------------------------
    # 9. CULTURE-SPECIFIC VALUE
    # --------------------------------------------------------

    print("\n[7] is_culture_specific")

    bad_culture_values = []

    for i, entry in enumerate(final, start=1):

        value = entry.get("is_culture_specific")

        if value not in VALID_CULTURE_VALUES:
            bad_culture_values.append(
                (i, entry.get("entry_id"), value)
            )

    if not bad_culture_values:
        print("  ✓ All values are yes/no/borderline.")
    else:
        errors.append(
            f"{len(bad_culture_values)} invalid "
            f"is_culture_specific values."
        )
        print(
            f"  ✗ {len(bad_culture_values)} invalid values."
        )

    # --------------------------------------------------------
    # 10. QUALITY SCORES
    # --------------------------------------------------------

    print("\n[8] QUALITY SCORES")

    score_errors = []

    for i, entry in enumerate(final, start=1):

        culture_status = entry.get("is_culture_specific")

        target_score = entry.get("target_word_correct")
        gloss_score = entry.get("gloss_accuracy")

        # For yes/borderline, scores MUST be 1-5
        if culture_status in {"yes", "borderline"}:

            if target_score not in {1, 2, 3, 4, 5}:
                score_errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"invalid target_word_correct={target_score}"
                )

            if gloss_score not in {1, 2, 3, 4, 5}:
                score_errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"invalid gloss_accuracy={gloss_score}"
                )

        # For no, scores may be blank
        elif culture_status == "no":

            if target_score not in {"", None}:
                score_errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"target_word_correct should be blank for 'no'"
                )

            if gloss_score not in {"", None}:
                score_errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"gloss_accuracy should be blank for 'no'"
                )

    if not score_errors:
        print("  ✓ All quality scores follow the README rules.")
    else:
        errors.extend(score_errors)
        print(f"  ✗ {len(score_errors)} score errors found.")

    # --------------------------------------------------------
    # 11. ASPECT CORRECT
    # --------------------------------------------------------

    print("\n[9] aspect_correct")

    aspect_errors = []

    for i, entry in enumerate(final, start=1):

        value = entry.get("aspect_correct")

        if value not in VALID_ASPECT_CORRECT:
            aspect_errors.append(
                f"Row {i}, entry_id={entry.get('entry_id')}: "
                f"invalid aspect_correct={value}"
            )

    if not aspect_errors:
        print("  ✓ aspect_correct is valid for every row.")
    else:
        errors.extend(aspect_errors)
        print(f"  ✗ {len(aspect_errors)} aspect errors.")

    # --------------------------------------------------------
    # 12. SUGGESTED ASPECT
    # --------------------------------------------------------

    print("\n[10] suggested_aspect")

    suggested_aspect_errors = []

    for i, entry in enumerate(final, start=1):

        aspect_correct = entry.get("aspect_correct")
        suggested = entry.get("suggested_aspect")

        if aspect_correct == "no":

            if suggested not in VALID_ASPECT_VALUES:

                suggested_aspect_errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"suggested_aspect required/invalid: {suggested}"
                )

        elif aspect_correct == "yes":

            if suggested not in {"", None}:

                suggested_aspect_errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"suggested_aspect must be blank when "
                    f"aspect_correct=yes"
                )

    if not suggested_aspect_errors:
        print("  ✓ suggested_aspect follows the README rules.")
    else:
        errors.extend(suggested_aspect_errors)
        print(
            f"  ✗ {len(suggested_aspect_errors)} "
            f"suggested_aspect errors."
        )

    # --------------------------------------------------------
    # 13. SUGGESTED TARGET
    # --------------------------------------------------------

    print("\n[11] suggested_target")

    suggested_target_errors = []

    for i, entry in enumerate(final, start=1):

        score = entry.get("target_word_correct")
        suggested = entry.get("suggested_target")

        if score in {1, 2}:

            if is_blank(suggested):

                suggested_target_errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"suggested_target required for score {score}"
                )

        elif score in {3, 4, 5}:

            if suggested not in {"", None}:

                suggested_target_errors.append(
                    f"Row {i}, entry_id={entry.get('entry_id')}: "
                    f"suggested_target must be blank for score {score}"
                )

    if not suggested_target_errors:
        print("  ✓ suggested_target follows the README rules.")
    else:
        errors.extend(suggested_target_errors)
        print(
            f"  ✗ {len(suggested_target_errors)} "
            f"suggested_target errors."
        )

    # --------------------------------------------------------
    # 14. COMMENTS
    # --------------------------------------------------------

    print("\n[12] COMMENTS")

    # We do NOT require comments for every row.
    # README says comments are especially useful for:
    # borderline/no/corrupted entries.
    # This is therefore reported, not treated as a hard failure.

    borderline_without_comments = 0
    no_without_comments = 0

    for entry in final:

        status = entry.get("is_culture_specific")
        comments = entry.get("comments")

        if is_blank(comments):

            if status == "borderline":
                borderline_without_comments += 1

            elif status == "no":
                no_without_comments += 1

    print(
        f"  ℹ Borderline entries without comments: "
        f"{borderline_without_comments}"
    )

    print(
        f"  ℹ 'no' entries without comments: "
        f"{no_without_comments}"
    )

    print("  ✓ Comments are not treated as mandatory for every row.")

    # --------------------------------------------------------
    # 15. FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if errors:

        print("FINAL RESULT: ✗ FAIL")
        print("=" * 70)

        print(f"\nTotal errors: {len(errors)}")

        # Show all errors
        for error in errors:
            print(f"  ✗ {error}")

        print("\nPlease fix the errors before submission.")

    else:

        print("FINAL RESULT: ✓ PASS")
        print("=" * 70)

        print("""
All required structural and annotation validations passed.

✓ 808 entries
✓ Same entry IDs
✓ Same original order
✓ No duplicate IDs
✓ Original source data unchanged
✓ All 8 annotation fields present
✓ Annotator ID consistent
✓ is_culture_specific valid
✓ Quality scores valid
✓ aspect_correct valid
✓ suggested_aspect rules valid
✓ suggested_target rules valid
✓ JSON structure valid

The file passes the automated README consistency check.
""")

    print("=" * 70)


    # --------------------------------------------------------
    # 15. INCOMPLETE ANNOTATION DIAGNOSTIC
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INCOMPLETE ANNOTATION DIAGNOSTIC")
    print("=" * 70)

    incomplete = []

    for i, entry in enumerate(final, start=1):
        culture = entry.get("is_culture_specific")
        aspect = entry.get("aspect_correct")

        if culture in ("", None) or aspect in ("", None):
            incomplete.append({
                "row": i,
                "entry_id": entry.get("entry_id"),
                "is_culture_specific": culture,
                "target_word_correct": entry.get("target_word_correct"),
                "gloss_accuracy": entry.get("gloss_accuracy"),
                "aspect_correct": aspect,
                "suggested_aspect": entry.get("suggested_aspect"),
                "suggested_target": entry.get("suggested_target"),
                "comments": entry.get("comments"),
            })

    print(f"\nIncomplete rows: {len(incomplete)}\n")

    for item in incomplete:
        print(
            f"Row {item['row']} | "
            f"ID {item['entry_id']} | "
            f"culture={repr(item['is_culture_specific'])} | "
            f"target={repr(item['target_word_correct'])} | "
            f"gloss={repr(item['gloss_accuracy'])} | "
            f"aspect={repr(item['aspect_correct'])}"
        )


if __name__ == "__main__":
    validate()