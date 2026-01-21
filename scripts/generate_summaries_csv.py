import json
import csv
import re
from pathlib import Path

# -------- CONFIG --------
ROOTS = [
    Path("gemini_judge_results_basic"),
    Path("gemini_judge_results_full"),
]
OUTPUT_CSV = "all_judgements_meta.csv"

META_COLS = [
    "paper_id",
    "judge_model",
    "judge_prompt",
    "generator_model",
    "summary_style",
]

# -------- HELPERS --------
def parse_judge_folder(name: str):
    """
    gemini_judge_results_basic -> (judge_model=gemini, judge_prompt=basic)
    """
    m = re.match(r"(.*)_judge_results_(.*)", name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_results_folder(name: str):
    """
    results_anthropic_long -> (generator_model=anthropic, summary_style=long)
    """
    parts = name.replace("results_", "").split("_")
    if len(parts) < 2:
        return None, None
    return parts[0], "_".join(parts[1:])


def flatten_json(d, parent_key="", sep="."):
    """
    Flattens nested dicts for CSV friendliness
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep))
        else:
            items[new_key] = v
    return items


# -------- MAIN --------
rows = []
score_keys = set()

for judge_dir in ROOTS:
    if not judge_dir.exists():
        print(f"Skipping missing folder: {judge_dir}")
        continue

    judge_model, judge_prompt = parse_judge_folder(judge_dir.name)
    if judge_model is None:
        continue

    for results_dir in judge_dir.iterdir():
        if not results_dir.is_dir():
            continue

        generator_model, summary_style = parse_results_folder(results_dir.name)
        if generator_model is None:
            continue

        for json_file in results_dir.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            flat = flatten_json(data)

            row = {
                "paper_id": json_file.stem.replace("_judge", ""),
                "judge_model": judge_model,
                "judge_prompt": judge_prompt,
                "generator_model": generator_model,
                "summary_style": summary_style,
            }

            # keep only *.score fields
            for k, v in flat.items():
                if k.endswith(".score"):
                    row[k] = v
                    score_keys.add(k)

            rows.append(row)

# -------- WRITE CSV (ONCE) --------
if not rows:
    raise RuntimeError("No rows found — check folder structure")

fieldnames = META_COLS + sorted(score_keys)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f" Wrote {len(rows)} rows → {OUTPUT_CSV}")
