from loader import load_dataset
from prompt_builder import load_prompt_template, build_annotation_prompt


dataset = load_dataset("data/culture_terms.json")

prompt_template = load_prompt_template(
    "prompts/annotation_prompt.txt"
)

prompt = build_annotation_prompt(
    dataset[0],
    prompt_template,
    "HUMAN_ANNOTATOR_01"
)

print(prompt)