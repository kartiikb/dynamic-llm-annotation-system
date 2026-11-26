#!/usr/bin/env python3
"""
Fill the alignment-check file (E5_alignment_check_telugu_*.json) with DeepSeek.
Uses the same .env as fill_with_deepseek.py (DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL).

Usage:
    python fill_alignment_check.py --input-dir . --out-dir filled --limit 5     # test
    python fill_alignment_check.py --input-dir . --out-dir filled               # full run
    python fill_alignment_check.py --input-dir . --out-dir filled --dry-run     # no API calls

Fills only: match, comments. Nothing else is touched (checker_id etc.).
`comments` is optional in the guide, so it is left EMPTY unless you pass --comments.
Resumable: rows that already have `match` are skipped on re-run.
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
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

SYSTEM = """You are a bilingual Telugu-English expert on the Ramayana checking sentence alignment.
Each row has a Telugu source segment (source_text, a Telugu prose retelling of the Ramayana; it may contain OCR noise or garbled characters) and the English sentence
an automatic aligner matched to it (aligned_english, from Hari Prasad Shastri's English translation of Valmiki's Ramayana, also with OCR typos).
Decide whether the English sentence expresses the same content as the Telugu segment. The English is a published literary translation of the Sanskrit epic, not a translation
of this exact text, so wording will differ: judge CONTENT (who does what, what is described), not wording.

match:
  "same"      = the English sentence expresses the same content (it may be freer or add a little)
  "partial"   = it shares part of the content (same event but a different part of it, or only half of the segment)
  "different" = it describes something else.

Be strict: sharing only a character name (e.g. both mention Rama or Ravana) is NOT enough for "partial"; the event/action must overlap.
If the source is too garbled to tell but the English is clearly about another event, use "different".

Reply with ONE JSON object only:
{"match": "same"|"partial"|"different", "comments": "<optional, may be empty>"}"""

COMMENT_STYLE = """
For "comments": only if something is genuinely useful, write one very short plain note (2-8 words, casual, no formal sentences, no "X, not Y" pattern). Otherwise "".
"""


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def make_validator(allow_comments):
    def validate(obj, dry=False):
        if dry:
            return {"match": "different", "comments": ""}
        m = str(obj.get("match", "")).strip().lower()
        if m not in ("same", "partial", "different"):
            raise ValueError(f"bad match: {obj}")
        c = str(obj.get("comments", "") or "").strip() if allow_comments else ""
        return {"match": m, "comments": c}
    return validate


def call_llm(system, user, validate, retries=6, dry_run=False):
    if dry_run:
        return validate(None, dry=True)
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "temperature": 0,
                      "response_format": {"type": "json_object"}},
                timeout=120,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return validate(parse_json(r.json()["choices"][0]["message"]["content"]))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed after {retries} attempts: {last}")


def fill_row(row, allow_comments, dry_run):
    shown = {k: row[k] for k in ("source_text", "aligned_english")}
    system = SYSTEM + (COMMENT_STYLE if allow_comments else "")
    user = "Judge this row. Return JSON.\n" + json.dumps(shown, ensure_ascii=False, indent=1)
    return call_llm(system, user, make_validator(allow_comments), dry_run=dry_run)


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def run_file(src, out, workers, limit, allow_comments, dry_run):
    data = json.loads((out if out.exists() else src).read_text(encoding="utf-8"))
    todo = [i for i, r in enumerate(data) if not str(r.get("match", "")).strip()]
    if limit:
        todo = todo[:limit]
    print(f"{src.name}: {len(data)} rows, {len(todo)} to fill", flush=True)
    lock, done, errs = threading.Lock(), [0], [0]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fill_row, data[i], allow_comments, dry_run): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                res = fut.result()
                with lock:
                    data[i]["match"] = res["match"]
                    data[i]["comments"] = res["comments"]
                    done[0] += 1
                    if done[0] % 10 == 0:
                        atomic_write(out, data)
                    print(f"  [{done[0]}/{len(todo)}] {data[i]['check_id']}", flush=True)
            except Exception as e:  # noqa: BLE001
                errs[0] += 1
                print(f"  ! {data[i]['check_id']} failed: {e}", file=sys.stderr, flush=True)

    atomic_write(out, data)
    print(f"wrote {out} ({done[0]} filled, {errs[0]} failed; re-run to retry failures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--out-dir", default="filled")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--comments", action="store_true", help="allow short optional comments (default: leave empty)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--files", nargs="*", help="file names (default: *alignment_check*.json)")
    args = ap.parse_args()

    if not args.dry_run and not API_KEY:
        sys.exit("DEEPSEEK_API_KEY is not set (put it in .env)")
    in_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    files = [in_dir / f for f in args.files] if args.files else sorted(in_dir.glob("*alignment_check*.json"))
    if not files:
        sys.exit(f"no *alignment_check*.json in {in_dir}")
    print(f"model={MODEL} base={BASE_URL} workers={args.workers}\n")
    for f in files:
        run_file(f, out_dir / f.name, args.workers, args.limit, args.comments, args.dry_run)


if __name__ == "__main__":
    main()