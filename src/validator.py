def validate_annotation(result, config):
    """
    Validate one LLM annotation against the configuration
    and the conditional rules from the annotation guide.
    """

    errors = []

    required_fields = [
        "annotator_id",
        "is_culture_specific",
        "target_word_correct",
        "gloss_accuracy",
        "aspect_correct",
        "suggested_aspect",
        "suggested_target",
        "comments"
    ]

    # Check all fields exist
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing field: {field}")

    if errors:
        return False, errors

    output_config = config["output_fields"]

    # --------------------------------------------------
    # Check annotator ID
    # --------------------------------------------------

    expected_annotator = config["annotator_id"]

    if result["annotator_id"] != expected_annotator:
        errors.append(
            f"Invalid annotator_id: {result['annotator_id']}"
        )

    # --------------------------------------------------
    # is_culture_specific
    # --------------------------------------------------

    allowed_culture = output_config[
        "is_culture_specific"
    ]["allowed_values"]

    if result["is_culture_specific"] not in allowed_culture:
        errors.append(
            f"Invalid is_culture_specific: "
            f"{result['is_culture_specific']}"
        )

    # --------------------------------------------------
    # aspect_correct
    # --------------------------------------------------

    if result["aspect_correct"] not in ["yes", "no"]:
        errors.append(
            f"Invalid aspect_correct: "
            f"{result['aspect_correct']}"
        )

    # --------------------------------------------------
    # Conditional rules
    # --------------------------------------------------

    culture_status = result["is_culture_specific"]

    if culture_status in ["yes", "borderline"]:

        # target_word_correct must be 1-5
        if result["target_word_correct"] not in [1, 2, 3, 4, 5]:
            errors.append(
                "target_word_correct must be 1-5 "
                "for yes/borderline entries."
            )

        # gloss_accuracy must be 1-5
        if result["gloss_accuracy"] not in [1, 2, 3, 4, 5]:
            errors.append(
                "gloss_accuracy must be 1-5 "
                "for yes/borderline entries."
            )

    elif culture_status == "no":

        # These may be blank
        if result["target_word_correct"] not in ["", None]:
            errors.append(
                "target_word_correct should be blank "
                "when is_culture_specific is no."
            )

        if result["gloss_accuracy"] not in ["", None]:
            errors.append(
                "gloss_accuracy should be blank "
                "when is_culture_specific is no."
            )

    # --------------------------------------------------
    # suggested_target
    # --------------------------------------------------

    target_score = result["target_word_correct"]

    if target_score in [1, 2]:

        if not result["suggested_target"].strip():
            errors.append(
                "suggested_target is required when "
                "target_word_correct is 1 or 2."
            )

    elif target_score in [3, 4, 5]:

        if result["suggested_target"] not in ["", None]:
            errors.append(
                "suggested_target must be blank when "
                "target_word_correct is 3, 4, or 5."
            )

    # --------------------------------------------------
    # suggested_aspect
    # --------------------------------------------------

    valid_aspects = output_config[
        "suggested_aspect"
    ]["allowed_values"]

    if result["aspect_correct"] == "no":

        if result["suggested_aspect"] not in valid_aspects:
            errors.append(
                "suggested_aspect is required and must "
                "be a valid Nida category."
            )

    elif result["aspect_correct"] == "yes":

        if result["suggested_aspect"] not in ["", None]:
            errors.append(
                "suggested_aspect must be blank when "
                "aspect_correct is yes."
            )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    if errors:
        return False, errors

    return True, []