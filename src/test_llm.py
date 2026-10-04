import json

from loader import load_dataset
from prompt_builder import load_prompt_template, build_annotation_prompt
from annotator import LLMAnnotator
from validator import validate_annotation


# Load configuration
with open(
    "config/dictionary_validation_config.json",
    "r",
    encoding="utf-8"
) as f:
    config = json.load(f)


# Load dataset
dataset = load_dataset("data/culture_terms.json")


# Load prompt
prompt_template = load_prompt_template(
    "prompts/annotation_prompt.txt"
)


# Test only the first entry
entry = dataset[0]


# Build prompt
prompt = build_annotation_prompt(
    entry,
    prompt_template,
    config["annotator_id"]
)


# Call LLM
annotator = LLMAnnotator()

result = annotator.annotate(prompt)


# Validate result
is_valid, errors = validate_annotation(
    result,
    config
)


print("\nLLM RESULT:")
print(json.dumps(
    result,
    ensure_ascii=False,
    indent=2
))


print("\nVALIDATION RESULT:")

if is_valid:
    print("✓ Annotation is valid")
else:
    print("✗ Annotation is invalid")

    for error in errors:
        print(f"  - {error}")