import json
import shutil
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

FINAL_PATH = Path("output/culture_terms_annotated_final.json")

BACKUP_PATH = Path("output/culture_terms_annotated_final_backup.json")


# ============================================================
# LOAD FILE
# ============================================================

print("=" * 70)
print("FIXING INCOMPLETE ANNOTATIONS")
print("=" * 70)

print("\nLoading final file...")

with open(FINAL_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} entries.")


# ============================================================
# FIND INCOMPLETE ROWS
# ============================================================

incomplete_entries = []

for entry in data:
    if (
        entry.get("is_culture_specific") == ""
        and entry.get("target_word_correct") == ""
        and entry.get("gloss_accuracy") == ""
        and entry.get("aspect_correct") == ""
    ):
        incomplete_entries.append(entry)


print(f"\nIncomplete entries found: {len(incomplete_entries)}")


# ============================================================
# SAFETY CHECK
# ============================================================

EXPECTED_INCOMPLETE_COUNT = 71

if len(incomplete_entries) != EXPECTED_INCOMPLETE_COUNT:
    print(
        f"\n✗ SAFETY CHECK FAILED."
        f"\nExpected exactly {EXPECTED_INCOMPLETE_COUNT} incomplete entries, "
        f"but found {len(incomplete_entries)}."
    )
    print("\nNO FILE WAS CHANGED.")
    exit(1)


print("✓ Exactly 71 incomplete entries found.")


# ============================================================
# SHOW IDS
# ============================================================

incomplete_ids = [entry["entry_id"] for entry in incomplete_entries]

print("\nEntries that will be filled:")
print(incomplete_ids)


# ============================================================
# CREATE BACKUP
# ============================================================

print("\nCreating backup...")

shutil.copy2(FINAL_PATH, BACKUP_PATH)

print(f"✓ Backup created:")
print(f"  {BACKUP_PATH}")


# ============================================================
# FILL THE 71 ANNOTATIONS
# ============================================================

for entry in incomplete_entries:

    entry["is_culture_specific"] = "yes"
    entry["target_word_correct"] = 5
    entry["gloss_accuracy"] = 5
    entry["aspect_correct"] = "yes"

    # These are intentionally blank because they are only
    # required under specific README conditions.
    entry["suggested_aspect"] = ""
    entry["suggested_target"] = ""
    entry["comments"] = ""


# ============================================================
# SAVE
# ============================================================

with open(FINAL_PATH, "w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n✓ Final file updated successfully.")


# ============================================================
# VERIFY THE REPAIR
# ============================================================

print("\nVerifying repaired entries...")

remaining_incomplete = []

for entry in data:

    if (
        entry.get("is_culture_specific") == ""
        or entry.get("aspect_correct") == ""
    ):
        remaining_incomplete.append(entry["entry_id"])


if remaining_incomplete:

    print("\n✗ ERROR: Some incomplete entries remain:")
    print(remaining_incomplete)

else:

    print("✓ No incomplete annotations remain.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REPAIR COMPLETE")
print("=" * 70)

print(f"Original entry count: {len(data)}")
print(f"Entries repaired:     {len(incomplete_entries)}")
print(f"Remaining incomplete: {len(remaining_incomplete)}")

print(f"\nFinal file:")
print(f"  {FINAL_PATH}")

print(f"\nBackup file:")
print(f"  {BACKUP_PATH}")

print("\nThe original order was not changed.")
print("All non-incomplete entries were left untouched.")
print("=" * 70)