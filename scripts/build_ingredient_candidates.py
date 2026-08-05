"""Create a reviewable ingredient-level candidate table from the legacy medicine CSV.

This script does not create recommendations.  A use listed for a combination product
is preserved as composition-level evidence and must be clinically/NZ verified later.
"""

from collections import Counter, defaultdict
from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "pipeline" / "legacy" / "Medicine_Details.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "pipeline" / "candidates"
OUTPUT_PATH = OUTPUT_DIR / "ingredient_candidates.csv"
REPORT_PATH = OUTPUT_DIR / "ingredient_candidate_report.json"

RISK_TERMS = {
    "antibiotic_or_antimicrobial": ("bacterial infection", "antibiotic", "antiviral"),
    "injection_or_parenteral": ("injection", "injectable", "vial", "ampoule"),
    "chronic_or_specialist_condition": (
        "diabetes",
        "hypertension",
        "schizophrenia",
        "mania",
        "heart failure",
        "cancer",
        "epilepsy",
    ),
}


def clean(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def ingredient_parts(composition: str) -> list[str]:
    return [
        part.split("(", 1)[0].strip().casefold()
        for part in composition.split("+")
        if part.split("(", 1)[0].strip()
    ]


def risk_flags(name: str, composition: str, uses: str) -> list[str]:
    record_text = f"{name} {composition} {uses}".casefold()
    return [
        label for label, terms in RISK_TERMS.items() if any(term in record_text for term in terms)
    ]


def sample(values: set[str], limit: int = 5) -> str:
    return " | ".join(sorted(values)[:limit])


def evidence_summary(values: Counter[str], limit: int = 10) -> str:
    return " | ".join(
        value
        for value, _ in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
    )


print("Reading legacy medicine source...")
source = pd.read_csv(SOURCE_PATH)
aggregates: dict[str, dict] = defaultdict(
    lambda: {
        "medicine_names": set(),
        "compositions": set(),
        "uses": Counter(),
        "risk_flags": set(),
    }
)

for _, row in source.iterrows():
    medicine_name = clean(row["Medicine Name"])
    composition = clean(row["Composition"])
    uses = clean(row["Uses"])
    flags = risk_flags(medicine_name, composition, uses)
    for ingredient in ingredient_parts(composition):
        item = aggregates[ingredient]
        item["medicine_names"].add(medicine_name)
        item["compositions"].add(composition)
        if uses:
            item["uses"][uses] += 1
        item["risk_flags"].update(flags)

rows = []
for ingredient, item in aggregates.items():
    rows.append(
        {
            "ingredient": ingredient,
            "source_medicine_count": len(item["medicine_names"]),
            "source_composition_count": len(item["compositions"]),
            "source_evidence": evidence_summary(item["uses"]),
            "sample_source_medicines": sample(item["medicine_names"]),
            "risk_flags": " | ".join(sorted(item["risk_flags"])),
            "relation_status": "composition_level_evidence_only",
            "review_status": "review_required",
        }
    )

result = pd.DataFrame(rows).sort_values(
    ["source_medicine_count", "ingredient"], ascending=[False, True]
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

report = {
    "source_row_count": int(len(source)),
    "unique_composition_count": int(source["Composition"].map(clean).nunique()),
    "candidate_ingredient_count": int(len(result)),
    "flagged_candidate_counts": {
        label: int(result["risk_flags"].str.contains(label, regex=False).sum())
        for label in RISK_TERMS
    },
    "output_note": (
        "This is an intermediate candidate table. It contains no NZ classification "
        "or approved symptom-to-ingredient recommendation."
    ),
}
REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Wrote {len(result)} ingredient candidates to {OUTPUT_PATH}")
