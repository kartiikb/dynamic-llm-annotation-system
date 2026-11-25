#!/usr/bin/env python3
"""
Fill the MIDAS review_*.json and template_check_*.json files with DeepSeek.

Setup:
    pip install requests python-dotenv        # python-dotenv is optional
    .env  ->  DEEPSEEK_API_KEY=...  DEEPSEEK_BASE_URL=https://api.deepseek.com  DEEPSEEK_MODEL=deepseek-flash

Usage:
    python fill_with_deepseek.py --input-dir . --out-dir filled
    python fill_with_deepseek.py --input-dir . --out-dir filled --limit 5      # quick test
    python fill_with_deepseek.py --input-dir . --out-dir filled --dry-run      # no API calls

Behaviour:
  * Originals are never modified; filled copies go to --out-dir under the same file names.
  * Resumable: rows/items already filled in the output file are skipped on re-run.
  * No marker is added to reviewer_note (set AI_TAG below if you want one).
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# --------------------------------------------------------------------------- config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # minimal .env parser fallback
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

AI_TAG = ""  # set to e.g. "[AI draft]" to mark AI-filled notes

# --------------------------------------------------------------------------- prompts
REVIEW_SYSTEM = """You are an expert native-level reviewer of Hindi and Telugu who checks answers to factual questions.
Each row has: relation, a question (in Hindi or Telugu, subject often in English), the model's answer, and the accepted
names of the correct (gold) entity in the question language and in English. The accepted names can be incomplete.

Decide which ONE rule applies:
R0 (wrong): the answer names a different entity, or is garbled / not a name
R1 (correct): the answer names the gold entity in the question's language, including spelling and transliteration variants, also when extra text follows the name
R2 (correct): the answer names the gold entity recognizably in another language or script (including mixed scripts)
R3 (wrong): the answer names a broader, narrower or only related entity (e.g. the country instead of the province), or describes the answer without naming it
R4 (wrong): the answer is right for a different entity that shares the subject's name (another film / place / person with the same name); the subject is fixed
R5 (correct): the answer is verifiably true for the relation at an earlier time but missing from the gold set
R6 (abstain): the answer gives no entity: a refusal, "couldn't find information", "Not specified"/"Not applicable", an echo of the whole question, a statement that the question has no answer

Conventions:
- An answer that only names the question's subject is R0.
- Judge only which entity the answer names. Grammar, honorifics, case endings, extra words after the name, language and script do not matter.
- Spelling/transliteration variants of a gold name count as the gold entity (R1, or R2 if another script).
- If the answer names an entity that is correct for the relation today but is NOT among the accepted names, use R0 (wrong) and put "correct but not in gold" in reviewer_note. If it was correct at an earlier time (e.g. a former capital), use R5.
- If the answer names two or more entities, it is correct only if the FIRST one it names is the gold entity.
- Relation meaning: "capital" = the capital of the subject; "capital of" = the entity (country/province/district...) of which the subject is the capital; "country" = country the subject is in; "place of birth" = city of birth.
- If unsure which rule applies, choose the more likely one and write "unsure" in reviewer_note.
- Matching the gold list is strong evidence, but use your own knowledge for transliteration variants not in the list.

Reply with ONE JSON object only:
{"rule": "R0".."R6", "human_label": "correct"|"wrong"|"abstain", "reviewer_note": "<short, may be empty>"}
rule->label mapping: R0,R3,R4 = wrong; R1,R2,R5 = correct; R6 = abstain."""

TEMPLATE_SYSTEM = """You are a native-level linguist of {language} checking texts used in a research experiment that asks a language model
factual questions. For the ONE item you are given, judge the {language} 'text' against the English original and the 'check' instruction.

Fields to return:
- verdict: "ok" (correct and natural), "minor" (understandable and means the right thing, but unnatural or slightly off: spelling, word choice, grammar) or "wrong" (asks/means something else, ambiguous, or hard to understand).
- rating: integer 1-5 (5 = exactly how a native speaker would write it, 3 = understandable but clearly unnatural, 1 = wrong or unintelligible).
- suggested_fix: your corrected version in the SAME FORM as 'text' (a string for a string; for objects/lists such as the phrase lists or worked example, the whole corrected object), or "" if nothing should change.
- reviewer_note: short optional comment (may be empty).

Notes: {{subj}} and {{question}} are placeholders and must be kept as-is. For no_answer_phrases the lists are used for substring/exact matching on lowercase answers
(English ones are 'I don't know'-style); say which phrases to remove or add. For honorifics, titles are stripped from person names before comparison.

Reply with ONE JSON object only:
{{"verdict": "...", "rating": 1-5, "suggested_fix": ..., "reviewer_note": "..."}}"""

RULE_TO_LABEL = {"R0": "wrong", "R1": "correct", "R2": "correct", "R3": "wrong",
                 "R4": "wrong", "R5": "correct", "R6": "abstain"}

# --------------------------------------------------------------------------- LLM call
def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def call_llm(system: str, user: str, validate, retries: int = 6, dry_run: bool = False):
    """Call DeepSeek, parse + validate JSON, retry on API/parse/validation errors."""
    if dry_run:
        return validate(None, dry=True)
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return validate(parse_json(content))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed after {retries} attempts: {last}")

# --------------------------------------------------------------------------- review rows
def validate_review(obj, dry=False):
    if dry:
        return {"human_label": "wrong", "rule": "R0", "reviewer_note": "dry run"}
    rule = str(obj.get("rule", "")).strip().upper()
    if rule not in RULE_TO_LABEL:
        raise ValueError(f"bad rule: {obj}")
    label = RULE_TO_LABEL[rule]  # rule decides the label, so they can never disagree
    note = str(obj.get("reviewer_note", "") or "").strip()
    return {"human_label": label, "rule": rule, "reviewer_note": note}


def review_user_prompt(row: dict) -> str:
    lang = "Hindi" if row.get("lang") == "hi" else "Telugu"
    gold_native = row.get("accepted_names_in_Hindi") or row.get("accepted_names_in_Telugu") or []
    payload = {
        "question_language": lang,
        "relation": row["relation"],
        "question": row["question"],
        "answer": row["answer"],
        f"accepted_names_in_{lang}": gold_native,
        "accepted_names_in_English": row.get("accepted_names_in_English", []),
    }
    return "Judge this row. Return JSON.\n" + json.dumps(payload, ensure_ascii=False, indent=1)


def fill_review_row(row: dict, dry_run: bool) -> dict:
    res = call_llm(REVIEW_SYSTEM, review_user_prompt(row), validate_review, dry_run=dry_run)
    note = res["reviewer_note"]
    return {"human_label": res["human_label"], "rule": res["rule"],
            "reviewer_note": f"{AI_TAG} {note}".strip()}

# --------------------------------------------------------------------------- template items
def validate_template(obj, dry=False):
    if dry:
        return {"verdict": "ok", "rating": 5, "suggested_fix": "", "reviewer_note": "dry run"}
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("ok", "minor", "wrong"):
        raise ValueError(f"bad verdict: {obj}")
    rating = int(obj.get("rating"))
    if not 1 <= rating <= 5:
        raise ValueError(f"bad rating: {obj}")
    fix = obj.get("suggested_fix", "")
    if fix is None:
        fix = ""
    return {"verdict": verdict, "rating": rating, "suggested_fix": fix,
            "reviewer_note": str(obj.get("reviewer_note", "") or "").strip()}


def fill_template_item(item: dict, language: str, dry_run: bool) -> dict:
    shown = {k: v for k, v in item.items()
             if k not in ("verdict", "rating", "suggested_fix", "reviewer_note")}
    system = TEMPLATE_SYSTEM.format(language=language)
    user = "Check this item. Return JSON.\n" + json.dumps(shown, ensure_ascii=False, indent=1)
    res = call_llm(system, user, validate_template, dry_run=dry_run)
    note = res["reviewer_note"]
    return {"verdict": res["verdict"], "rating": res["rating"],
            "suggested_fix": res["suggested_fix"],
            "reviewer_note": f"{AI_TAG} {note}".strip()}

# --------------------------------------------------------------------------- driver
def atomic_write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run_file(src: Path, out: Path, workers: int, limit: int | None, dry_run: bool):
    data = json.loads(src.read_text(encoding="utf-8"))
    if out.exists():  # resume from previous partial output
        prev = json.loads(out.read_text(encoding="utf-8"))
        data = prev
    is_review = isinstance(data, list)
    rows = data if is_review else data["items"]
    done_key = "human_label" if is_review else "verdict"
    language = None if is_review else data.get("language", "the target language")

    todo = [i for i, r in enumerate(rows) if not str(r.get(done_key, "")).strip()]
    if limit:
        todo = todo[:limit]
    print(f"{src.name}: {len(rows)} entries, {len(todo)} to fill", flush=True)

    lock = threading.Lock()
    counter = {"n": 0, "err": 0}

    def work(i):
        row = rows[i]
        return i, (fill_review_row(row, dry_run) if is_review
                   else fill_template_item(row, language, dry_run))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                _, res = fut.result()
                with lock:
                    rows[i].update(res)
                    counter["n"] += 1
                    if counter["n"] % 10 == 0:
                        atomic_write(out, data)
                    print(f"  [{counter['n']}/{len(todo)}] n={rows[i].get('n')} done", flush=True)
            except Exception as e:  # noqa: BLE001
                counter["err"] += 1
                print(f"  ! n={rows[i].get('n')} failed: {e}", file=sys.stderr, flush=True)

    atomic_write(out, data)
    print(f"{src.name}: wrote {out} ({counter['n']} filled, {counter['err']} failed; "
          f"re-run to retry failures)\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=".", help="folder with the 4 json files")
    ap.add_argument("--out-dir", default="filled")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only fill first N empty entries per file")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; fill with dummy values")
    ap.add_argument("--files", nargs="*", help="specific json file names (default: review_*.json, template_check_*.json)")
    args = ap.parse_args()

    if not args.dry_run and not API_KEY:
        sys.exit("DEEPSEEK_API_KEY is not set (put it in .env)")

    in_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    if args.files:
        files = [in_dir / f for f in args.files]
    else:
        files = sorted(in_dir.glob("*review_*.json")) + sorted(in_dir.glob("*template_check_*.json"))
        files = [f for f in dict.fromkeys(files)]
    if not files:
        sys.exit(f"no review_*.json / template_check_*.json found in {in_dir}")

    print(f"model={MODEL} base={BASE_URL} workers={args.workers}\n")
    for f in files:
        run_file(f, out_dir / f.name, args.workers, args.limit, args.dry_run)


if __name__ == "__main__":
    main()