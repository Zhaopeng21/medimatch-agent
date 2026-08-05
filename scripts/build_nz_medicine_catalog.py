"""Build the runtime NZ General Sale medicine catalog from local source snapshots.

Only ingredients that exactly match both the PHARMAC Schedule and a Medsafe
General Sale classification are included. Legacy uses are retained as traceable
retrieval evidence, not asserted as NZ clinical guidance.
"""

from datetime import date
from html.parser import HTMLParser
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED_PATH = PROJECT_ROOT / "data" / "pipeline" / "candidates" / "ingredient_candidates.csv"
PHARMAC_PATH = PROJECT_ROOT / "data" / "pipeline" / "sources" / "pharmac_schedule_2026-08.xml"
MEDSAFE_PATH = PROJECT_ROOT / "data" / "pipeline" / "sources" / "medsafe_classification_2026-08.html"
CATALOG_PATH = PROJECT_ROOT / "data" / "nz_medicine_catalog.json"
METADATA_PATH = PROJECT_ROOT / "data" / "nz_medicine_catalog_metadata.json"
VECTOR_PATH = PROJECT_ROOT / "data" / "nz_medicine_faiss"
PHARMAC_URL = "https://schedule.pharmac.govt.nz/pub/schedule/archive/Schedule_2026-08.xml"
MEDSAFE_URL = "https://www.medsafe.govt.nz/profs/class/classintro.asp"


class ClassificationTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_listing = False
        self.in_cell = False
        self.cell: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table" and dict(attrs).get("class") == "Listing":
            self.in_listing = True
        elif self.in_listing and tag == "tr":
            self.row = []
        elif self.in_listing and tag in {"td", "th"}:
            self.in_cell = True
            self.cell = []

    def handle_endtag(self, tag):
        if self.in_listing and tag in {"td", "th"}:
            self.row.append("".join(self.cell).strip())
            self.in_cell = False
        elif self.in_listing and tag == "tr" and len(self.row) == 3:
            self.rows.append(self.row)
        elif tag == "table" and self.in_listing:
            self.in_listing = False

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def parse_medsafe_general_sale() -> dict[str, list[str]]:
    parser = ClassificationTableParser()
    parser.feed(MEDSAFE_PATH.read_text(encoding="utf-8", errors="replace"))
    general_sale: dict[str, list[str]] = {}
    for ingredient, conditions, classification in parser.rows[1:]:
        if classification == "General Sale":
            general_sale.setdefault(normalized(ingredient), []).append(conditions)
    return general_sale


def parse_pharmac_products() -> dict[str, dict[str, set[str]]]:
    namespace = {"s": "http://schedule.pharmac.govt.nz/2006/07/Schedule#"}
    root = ET.parse(PHARMAC_PATH).getroot()
    products: dict[str, dict[str, set[str]]] = {}
    for chemical in root.findall(".//s:Chemical", namespace):
        name = chemical.findtext("s:Name", default="", namespaces=namespace).strip()
        if not name:
            continue
        item = products.setdefault(normalized(name), {"names": set(), "formulations": set()})
        item["names"].add(name)
        for formulation in chemical.findall("s:Formulation", namespace):
            formulation_name = formulation.findtext("s:Name", default="", namespaces=namespace).strip()
            if formulation_name:
                item["formulations"].add(formulation_name)
            for brand in formulation.findall("s:Brand", namespace):
                brand_name = brand.findtext("s:Name", default="", namespaces=namespace).strip()
                if brand_name:
                    item["names"].add(brand_name)
    return products


def main():
    candidates = pd.read_csv(DERIVED_PATH).fillna("")
    general_sale = parse_medsafe_general_sale()
    pharmac_products = parse_pharmac_products()
    catalog = []

    for _, row in candidates.iterrows():
        ingredient = str(row["ingredient"]).strip()
        key = normalized(ingredient)
        if row["risk_flags"] or key not in general_sale or key not in pharmac_products:
            continue
        nz_product = pharmac_products[key]
        catalog.append(
            {
                "active_ingredient": ingredient.title(),
                "nz_search_query": ingredient,
                "nz_classification": "General Sale",
                "classification_conditions": general_sale[key],
                "pharmac_product_examples": sorted(nz_product["names"])[:8],
                "pharmac_formulations": sorted(nz_product["formulations"])[:8],
                "legacy_treatment_evidence": str(row["source_evidence"]),
                "source_medicine_count": int(row["source_medicine_count"]),
                "eligibility": "general_sale_exact_nz_match",
                "source_urls": [PHARMAC_URL, MEDSAFE_URL],
                "last_verified": str(date.today()),
            }
        )

    catalog.sort(key=lambda item: item["active_ingredient"])
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "catalog_name": "MediMatch NZ General Sale Medicine Catalog",
                "record_count": len(catalog),
                "sources": {
                    "pharmac_schedule": PHARMAC_URL,
                    "medsafe_classification": MEDSAFE_URL,
                },
                "limitations": (
                    "General Sale classification can include formulation, strength, pack, or use conditions. "
                    "Legacy treatment evidence is retained for retrieval only and is not an NZ clinical guideline."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    documents = [
        Document(
            page_content=(
                f"Ingredient: {item['active_ingredient']}. "
                f"Treatment evidence: {item['legacy_treatment_evidence']}. "
                f"NZ classification: {item['nz_classification']}."
            ),
            metadata={
                "medicine_name": item["active_ingredient"],
                "treats": item["legacy_treatment_evidence"],
                "composition": item["active_ingredient"],
                "side_effects": "Read the NZ product label and ask a pharmacist if unsure.",
                "nz_search_query": item["nz_search_query"],
            },
        )
        for item in catalog
    ]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    FAISS.from_documents(documents, embeddings).save_local(str(VECTOR_PATH))
    print(f"Built {len(catalog)} NZ General Sale records at {CATALOG_PATH}")


if __name__ == "__main__":
    main()
