from typing import Any

SEVERITIES = {"none", "minor", "major", "critical", "not_applicable", "uncertain"}
STRATEGIES = {"RET", "EQV", "EXP", "GEN", "NEU", "SUB", "OMI", "DIS", "None"}
ANNOTATION_FIELDS = ["annotator_id", "severity", "strategy", "comments"]


def validate_annotation(result: dict[str, Any], annotator_id: str):
    errors = []
    if not isinstance(result, dict):
        return False, ["Annotation must be a JSON object."]

    for field in ANNOTATION_FIELDS:
        if field not in result:
            errors.append(f"Missing field: {field}")

    if errors:
        return False, errors

    if result["annotator_id"] != annotator_id:
        errors.append(f"annotator_id must be {annotator_id!r}.")

    severity = result["severity"]
    strategy = result["strategy"]
    comments = result["comments"]

    if severity not in SEVERITIES:
        errors.append(f"Invalid severity: {severity!r}.")
    if strategy not in STRATEGIES:
        errors.append(f"Invalid strategy: {strategy!r}.")
    if not isinstance(comments, str):
        errors.append("comments must be a string.")
    elif severity in {"major", "critical", "uncertain"} and not comments.strip():
        errors.append(f"comments is required when severity is {severity!r}.")

    if severity in {"not_applicable", "uncertain"}:
        if strategy != "None":
            errors.append("strategy must be 'None' for not_applicable or uncertain.")
    elif strategy == "None":
        errors.append("'None' strategy is allowed only with not_applicable or uncertain.")

    return (not errors), errors


def is_complete_valid_row(row: dict[str, Any], annotator_id: str) -> bool:
    annotation = {field: row.get(field) for field in ANNOTATION_FIELDS}
    ok, _ = validate_annotation(annotation, annotator_id)
    return ok
