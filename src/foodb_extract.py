"""
One-off streaming extraction of nutrient rows from FooDB (2020-04-07 JSON release).

FooDB's Content.json is ~3.5 GB of JSON Lines. We only need:
  * every row with source_type == "Nutrient"   (macronutrients: fat, protein, carbs, fibre, energy, ...)
  * rows for a short list of compounds that are nutritionally important
    (water, minerals, vitamins, sugars, cholesterol, caffeine, ethanol)

Output: data/interim/foodb_nutrient_content.csv  (~ a few MB), reused by d1_nutrient_pipeline.py.

Run once:  python3 src/foodb_extract.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOODB = ROOT / "foodb_2020_04_07_json"
OUT = ROOT / "data" / "interim"
OUT.mkdir(parents=True, exist_ok=True)

# FooDB compound ids (from Compound.json) for the nutritionally relevant compounds
NUTRIENT_COMPOUNDS = {
    31063: "water", 3524: "sodium", 3522: "potassium", 3514: "calcium", 16258: "iron",
    3521: "phosphorus", 3730: "zinc", 3583: "copper", 3637: "manganese", 13403: "selenium",
    13272: "cholesterol", 3716: "sugars", 1131: "sucrose", 12533: "glucose", 12531: "fructose",
    1145: "lactose", 1193: "maltose", 1128: "starch", 1223: "vitamin_c", 8425: "thiamine",
    12163: "riboflavin", 1014: "niacin", 574: "vitamin_b6", 14507: "folate", 13267: "vitamin_b12",
    13831: "retinol", 14616: "beta_carotene", 565: "vitamin_e", 125170: "vitamin_k",
    2100: "caffeine", 753: "ethanol",
}
KEEP = ["food_id", "source_type", "source_id", "orig_source_name", "orig_food_common_name",
        "orig_food_part", "preparation_type", "standard_content", "orig_content", "orig_unit",
        "citation"]


def main():
    nutrients = {}
    with open(FOODB / "Nutrient.json") as fh:
        for line in fh:
            d = json.loads(line)
            nutrients[d["id"]] = d["name"]

    n_in = n_out = 0
    with open(FOODB / "Content.json") as fh, open(OUT / "foodb_nutrient_content.csv", "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(KEEP + ["nutrient"])
        for line in fh:
            n_in += 1
            # cheap pre-filter before JSON parsing
            if '"source_type":"Nutrient"' in line:
                d = json.loads(line)
                name = nutrients.get(d["source_id"])
            elif '"source_type":"Compound"' in line:
                d = json.loads(line)
                name = NUTRIENT_COMPOUNDS.get(d["source_id"])
                if name is None:
                    continue
            else:
                continue
            w.writerow([d.get(k) for k in KEEP] + [name])
            n_out += 1
    print(f"scanned {n_in:,} content rows, kept {n_out:,} nutrient rows")


if __name__ == "__main__":
    main()
