from pathlib import Path


def load_prompt_template(prompt_path):
    """
    Load the annotation prompt template.
    """

    prompt_path = Path(prompt_path)

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}"
        )

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_annotation_prompt(entry, prompt_template, annotator_id):
    """
    Insert one dictionary entry into the annotation prompt.
    """

    entry_information = f"""
ANNOTATOR ID:
{annotator_id}

ENTRY ID:
{entry["entry_id"]}

SOURCE LANGUAGE:
{entry["source_language"]}

SOURCE WORD:
{entry["source_word"]}

TARGET LANGUAGE:
{entry["target_language"]}

TARGET WORD:
{entry["target_word"]}

CULTURE ASPECT:
{entry["culture_aspect"]}

GLOSS:
{entry["gloss"]}

SAMPLE:
{entry["sample"]}

KANDA:
{entry["kanda"]}
"""

    final_prompt = f"""
{prompt_template}

Now annotate the following dictionary entry:

{entry_information}

Return ONLY the JSON object specified in the output format.
"""

    return final_prompt