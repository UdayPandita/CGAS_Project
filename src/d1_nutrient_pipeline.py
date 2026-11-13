"""
Deliverable 1 (part B) — Ingredient × Nutrient matrix from FooDB
================================================================

Inputs
  data/interim/foodb_nutrient_content.csv   nutrient rows streamed out of FooDB Content.json
                                            (built by src/foodb_extract.py)
  data/mappings/ingredient_to_usda.csv      hand-curated map: recipe ingredient -> one FooDB
                                            source record (USDA SR description, or DTU for the
                                            few foods FooDB's USDA subset lacks)

Cleaning rules (all derived from checks documented in reports/D1_data_pipeline.md)
  * USDA rows are the primary source. Their "Fatty acids" field is USDA "Total lipid (fat)".
  * Energy: FooDB's standard_content sometimes stores kJ; orig_content always holds kcal.
  * DTU rows are used only for the 5 ingredients with no USDA record, and only for nutrients
    where DTU agrees with USDA on shared foods (median log10 ratio ~ 0). DTU cholesterol is off
    by ~500x and DTU energy mixes kJ/kcal, so cholesterol is imputed and energy is computed
    from Atwater factors (4 kcal/g protein, 4 kcal/g carbohydrate, 9 kcal/g fat).
  * Missing values: animal foods get 0 fibre and meat/fish/eggs get 0 sugars (structural zeros);
    anything else missing is filled with the median of the ingredient's category. Every
    category-median imputed cell is counted (n_imputed).

Output
  data/processed/ingredient_nutrients.csv   381 recipe ingredients; nutrients per 100 g + provenance
  data/processed/food_nutrient_catalog.csv  extended catalog: curated recipe ingredients + every other
                                            FooDB food with a usable profile (see build_catalog)
  results/d1/d1_nutrient_*.png, d1_nutrient_summary.json

Run:  python3 src/d1_nutrient_pipeline.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "data" / "interim" / "foodb_nutrient_content.csv"
MAPPING = ROOT / "data" / "mappings" / "ingredient_to_usda.csv"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results" / "d1"
RES.mkdir(parents=True, exist_ok=True)

# output column -> (FooDB nutrient label, divisor to convert FooDB mg/100 g to the output unit)
FEATURES = {
    "energy_kcal":     ("Energy", 1),
    "protein_g":       ("Proteins", 1000),
    "fat_g":           ("Fatty acids", 1000),     # USDA "Total lipid (fat)"
    "carbohydrate_g":  ("Carbohydrate", 1000),
    "fiber_g":         ("Fiber (dietary)", 1000),
    "sugars_g":        ("sugars", 1000),
    "alcohol_g":       ("ethanol", 1000),
    "cholesterol_mg":  ("cholesterol", 1),
    "sodium_mg":       ("sodium", 1),
    "potassium_mg":    ("potassium", 1),
    "calcium_mg":      ("calcium", 1),
    "iron_mg":         ("iron", 1),
    "phosphorus_mg":   ("phosphorus", 1),
    "zinc_mg":         ("zinc", 1),
}
DTU_TRUSTED = {"protein_g", "carbohydrate_g", "fiber_g", "sugars_g", "sodium_mg", "potassium_mg",
               "calcium_mg", "iron_mg", "phosphorus_mg", "zinc_mg"}
ANIMAL = {"meat", "fish/seafood", "dairy", "animal product"}


def load_content():
    c = pd.read_csv(CONTENT, low_memory=False)
    c["value"] = pd.to_numeric(c["standard_content"], errors="coerce")
    e = c["nutrient"] == "Energy"
    c.loc[e, "value"] = pd.to_numeric(c.loc[e, "orig_content"], errors="coerce")   # kcal
    return c


def profile(rows: pd.DataFrame, source: str) -> dict:
    """Median value of each feature over the matched source rows."""
    out = {}
    for col, (label, div) in FEATURES.items():
        if source == "DTU":
            if col == "fat_g":
                label = "Fat"                       # DTU 'Fatty acids' is not total fat
            elif col not in DTU_TRUSTED:
                out[col] = np.nan
                continue
        v = rows.loc[rows["nutrient"] == label, "value"].dropna()
        out[col] = float(v.median() / div) if len(v) else np.nan
    if source == "DTU":
        p, f, cb = out["protein_g"], out["fat_g"], out["carbohydrate_g"]
        out["energy_kcal"] = 4 * np.nan_to_num(p) + 9 * np.nan_to_num(f) + 4 * np.nan_to_num(cb)
        out["alcohol_g"] = 0.0
    return out


FOODB_DIR = ROOT / "foodb_2020_04_07_json"
# USDA product descriptions that are not representative of a basic food
# (upper-case brand names such as KRAFT / CAMPBELL'S are filtered separately)
_NOT_BASIC_CI = re.compile(r"babyfood|baby food|fast foods?|restaurant|school lunch|infant formula|"
                           r"meatless|imitation|frozen entree|pot pie", re.I)
STRUCTURAL_ZERO_GROUPS = {"Aquatic foods", "Animal foods", "Eggs"}     # no fibre, no sugar
# USDA records of regional/traditional foods; foods known only from these are flagged 'specialty'
SPECIALTY = re.compile(r"\((Alaska Native|Northern Plains Indians|Navajo|Hopi|Apache|Shoshone Bannock)\)", re.I)
EXCLUDE_FOODS = re.compile(r"^Other |^Baby food|^Unclassified", re.I)


def snake(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def build_catalog(nut: pd.DataFrame):
    """Extend the 344 curated recipe ingredients with every other FooDB food that has a
    usable nutrient profile (core macronutrients present).

    Profile of an auto-added food = median over its *basic* source products: branded
    items (upper-case brand names), fast food, baby food, imitation products and ready
    meals are skipped, and raw (or uncooked/dry) products are preferred when the food has any. USDA is used
    when available, otherwise DTU with the same trusted-nutrient rules as above."""
    foods = pd.read_json(FOODB_DIR / "Food.json", lines=True)[["id", "name", "food_group", "food_subgroup"]]
    foods["food_group"] = foods["food_group"].replace({"Herbs and spices": "Herbs and Spices"})
    c = load_content()
    c = c[c["citation"].isin(["USDA", "DTU"])]
    covered = set(nut["foodb_food_id"].dropna().astype(int))

    rows = []
    for fid, g in c.groupby("food_id"):
        if fid in covered:
            continue
        f = foods[foods["id"] == fid]
        if f.empty or EXCLUDE_FOODS.search(f["name"].iloc[0]):
            continue
        src = "USDA" if (g["citation"] == "USDA").any() else "DTU"
        g = g[g["citation"] == src]
        desc = g["orig_food_common_name"].fillna("")
        basic = ~desc.str.contains(_NOT_BASIC_CI) & ~desc.str.contains(r"\b[A-Z]{3,}\b")
        if basic.any():
            g, desc = g[basic], desc[basic]
        raw = desc.str.contains(r"\braw\b|\buncooked\b|\bdry\b", case=False)    # grains: "uncooked"/"dry"
        if raw.any():
            g = g[raw]
        p = profile(g, src)
        names = g["orig_food_common_name"].fillna("").unique()
        specialty = bool(len(names)) and all(SPECIALTY.search(x) for x in names)
        core = ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g"]
        if any(pd.isna(p[k]) for k in core):
            continue
        rows.append({"ingredient": snake(f["name"].iloc[0]), "display_name": f["name"].iloc[0],
                     "food_group": f["food_group"].iloc[0], "food_subgroup": f["food_subgroup"].iloc[0],
                     "origin": "foodb_auto", "source": src, "match_type": "auto",
                     "description": f"median of {g['orig_food_common_name'].nunique()} {src} product(s)",
                     "n_products": int(g["orig_food_common_name"].nunique()), "foodb_food_id": int(fid),
                     "specialty": specialty, **p})
    auto = pd.DataFrame(rows)
    auto = auto[~auto["ingredient"].isin(nut["ingredient"])]           # curated name wins on collision
    auto = auto.drop_duplicates("ingredient")

    feats = list(FEATURES)
    # structural zeros, then food-group medians (computed over curated + auto foods together)
    sz = auto["food_group"].isin(STRUCTURAL_ZERO_GROUPS)
    auto.loc[sz & auto["fiber_g"].isna(), "fiber_g"] = 0.0
    auto.loc[sz & auto["sugars_g"].isna(), "sugars_g"] = 0.0
    auto.loc[auto["alcohol_g"].isna() & (auto["food_group"] != "Beverages"), "alcohol_g"] = 0.0
    auto["n_imputed"] = auto[feats].isna().sum(axis=1)

    cur = nut[nut["match_type"] != "none"].copy()
    cur = cur.merge(foods.rename(columns={"id": "foodb_food_id"}), on="foodb_food_id", how="left",
                    suffixes=("", "_foodb"))
    cur["display_name"] = cur["ingredient"]
    cur["origin"] = "recipe_curated"
    cur["n_products"] = 1
    cur["specialty"] = False
    cat = pd.concat([cur, auto], ignore_index=True)
    for f in feats:
        med = cat.groupby("food_group")[f].transform("median")
        fill = cat[f].isna()
        cat.loc[fill, f] = med[fill]
        cat.loc[cat[f].isna(), f] = cat[f].median()
    cat[feats] = cat[feats].round(3)
    keep = ["ingredient", "display_name", "origin", "category", "food_group", "food_subgroup", "source",
            "description", "match_type", "n_products", "n_imputed", "foodb_food_id", "specialty"] + feats
    cat = cat[keep]
    macro = cat[["protein_g", "fat_g", "carbohydrate_g", "alcohol_g"]].sum(axis=1)
    bad = (macro > 105) | (cat["energy_kcal"] > 905)
    cat = cat[~bad].reset_index(drop=True)
    cat.to_csv(PROC / "food_nutrient_catalog.csv", index=False)

    summary = {"catalog_foods": int(len(cat)),
               "recipe_curated": int((cat["origin"] == "recipe_curated").sum()),
               "foodb_auto": int((cat["origin"] == "foodb_auto").sum()),
               "auto_source": cat.loc[cat["origin"] == "foodb_auto", "source"].value_counts().to_dict(),
               "dropped_failed_sanity": int(bad.sum()),
               "specialty_foods": int(cat["specialty"].sum()),
               "auto_cells_imputed_pct": float(auto["n_imputed"].sum() / (len(auto) * len(feats)) * 100),
               "food_groups": cat["food_group"].value_counts().to_dict()}
    return cat, summary


def main():
    ing = pd.read_csv(PROC / "ingredients.csv")
    ing = ing[ing["in_recipes"]][["ingredient", "category", "n_compounds", "recipe_freq"]]
    m = pd.read_csv(MAPPING).merge(ing, on="ingredient", how="right")
    c = load_content()
    wanted = set(zip(m["source"].dropna(), m["description"].dropna()))
    c = c[[k in wanted for k in zip(c["citation"], c["orig_food_common_name"])]]
    groups = {k: g for k, g in c.groupby(["citation", "orig_food_common_name"])}

    rows = []
    for _, r in m.iterrows():
        rec = {"ingredient": r["ingredient"], "category": r["category"], "source": r["source"],
               "description": r["description"], "match_type": r["match_type"]}
        if r["match_type"] != "none":
            g = groups[(r["source"], r["description"])]
            rec.update(profile(g, r["source"]))
            rec["foodb_food_id"] = int(g["food_id"].mode().iloc[0])
        rows.append(rec)
    nut = pd.DataFrame(rows)
    feats = list(FEATURES)
    has = nut["match_type"] != "none"

    # ---- imputation (only for matched ingredients)
    raw_missing = nut.loc[has, feats].isna()
    nut["n_imputed"] = 0
    nut.loc[has, "n_imputed"] = raw_missing.sum(axis=1)
    animal = nut["category"].isin(ANIMAL) & has
    nut.loc[animal & nut["fiber_g"].isna(), "fiber_g"] = 0.0
    nut.loc[animal & ~nut["category"].eq("dairy") & nut["sugars_g"].isna(), "sugars_g"] = 0.0
    nut.loc[has & nut["alcohol_g"].isna() & (nut["category"] != "alcoholic beverage"), "alcohol_g"] = 0.0
    # structural zeros above (no fibre in animal foods, no sugar in meat/fish, no alcohol
    # outside beverages) are not estimates; only category-median fills count as imputed
    nut.loc[has, "n_imputed"] = nut.loc[has, feats].isna().sum(axis=1)
    for f in feats:
        med = nut[has].groupby("category")[f].transform("median")
        fill = has & nut[f].isna()
        nut.loc[fill, f] = med[fill]
        nut.loc[has & nut[f].isna(), f] = nut.loc[has, f].median()       # category with no data at all
    nut[feats] = nut[feats].round(3)

    # ---- sanity checks
    macro = nut.loc[has, ["protein_g", "fat_g", "carbohydrate_g", "alcohol_g"]].sum(axis=1)
    assert (macro <= 105).all(), nut.loc[has][macro > 105]
    assert (nut.loc[has, "energy_kcal"] <= 905).all()
    atwater = 4 * nut["protein_g"] + 9 * nut["fat_g"] + 4 * nut["carbohydrate_g"] + 7 * nut["alcohol_g"]
    err = (nut.loc[has, "energy_kcal"] - atwater[has]).abs()
    nut["energy_check_kcal_diff"] = (nut["energy_kcal"] - atwater).round(1)

    nut.to_csv(PROC / "ingredient_nutrients.csv", index=False)
    catalog, cat_summary = build_catalog(nut)

    # ---- summary
    summary = {
        "extended_catalog": cat_summary,
        "recipe_ingredients": int(len(nut)),
        "with_nutrients": int(has.sum()),
        "match_type": nut["match_type"].value_counts().to_dict(),
        "source": nut.loc[has, "source"].value_counts().to_dict(),
        "recipe_mentions_covered_pct": float(nut.loc[has, "ingredient"].map(
            ing.set_index("ingredient")["recipe_freq"]).sum() / ing["recipe_freq"].sum() * 100),
        "cells_imputed_pct": float(raw_missing.values.mean() * 100),
        "imputed_by_feature_pct": (raw_missing.mean() * 100).round(1).to_dict(),
        "energy_vs_atwater_median_abs_diff_kcal": float(err.median()),
        "energy_vs_atwater_within_10pct": float((err <= 0.1 * nut.loc[has, "energy_kcal"].clip(lower=10)).mean() * 100),
        "no_match": nut.loc[~has, "ingredient"].tolist(),
        "examples": nut[nut["ingredient"].isin(["butter", "chicken", "olive_oil", "lemon_juice", "oregano",
                                                "egg", "rice", "whiskey"])].set_index("ingredient")[
            ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sodium_mg"]].to_dict("index"),
        "both_flavor_and_nutrient": int((has & nut["ingredient"].isin(ing.loc[ing["n_compounds"] >= 1, "ingredient"])).sum()),
    }
    (RES / "d1_nutrient_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---- figure: category nutrient profile + coverage
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})
    a = ax[0]
    z = nut[has].copy()
    zf = np.log1p(z[feats])
    zf = (zf - zf.mean()) / zf.std()
    H = zf.groupby(z["category"]).mean()
    im = a.imshow(H.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    a.set_xticks(range(len(feats))); a.set_xticklabels(feats, rotation=50, ha="right")
    a.set_yticks(range(len(H))); a.set_yticklabels(H.index)
    a.set_title("A. Mean nutrient profile per category (z-score of log values)")
    fig.colorbar(im, ax=a, fraction=0.03)
    for sp in a.spines.values():
        sp.set_visible(False)
    a = ax[1]
    ct = nut.groupby(["category", "match_type"]).size().unstack(fill_value=0)[["direct", "proxy", "none"]]
    ct = ct.loc[ct.sum(1).sort_values().index]
    left = np.zeros(len(ct))
    for col, color in zip(ct.columns, ["#2f6db3", "#8fb3de", "#d6d9dd"]):
        a.barh(ct.index, ct[col], left=left, color=color, label=col)
        left += ct[col].to_numpy()
    a.legend(frameon=False); a.set_xlabel("recipe ingredients")
    a.set_title("B. Nutrient match coverage")
    fig.tight_layout()
    fig.savefig(RES / "d1_nutrient_overview.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2, default=str))
    print(pd.DataFrame(summary["examples"]).T.to_string())


if __name__ == "__main__":
    main()
