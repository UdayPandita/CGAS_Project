"""
Command-line ingredient substitute finder — flavor or nutrient based.

Flavor mode (chemically similar; Jaccard on flavor compounds, with 95% CI and confidence):
  python3 src/recommend.py butter
  python3 src/recommend.py basil --k 10 --weighting idf
  python3 src/recommend.py lemon --same-category --min-confidence Medium

Nutrient mode (nutritionally similar; FooDB/USDA nutrients per 100 g):
  python3 src/recommend.py butter --mode nutrient
  python3 src/recommend.py butter --mode nutrient --lower fat
  python3 src/recommend.py beef   --mode nutrient --lower fat --higher protein
  python3 src/recommend.py rice   --mode nutrient --features macro
  python3 src/recommend.py jackfruit --mode nutrient              # any of 808 FooDB foods (default catalog)
  python3 src/recommend.py butter --mode nutrient --catalog recipe # suggest recipe ingredients only

  python3 src/recommend.py --list [--mode nutrient]     # ingredients that can be queried
"""
import argparse
import difflib
import re

import pandas as pd

from substitution_model import GOAL_ALIASES, FlavorSubstitutionModel, NutrientSubstitutionModel

p = argparse.ArgumentParser(description="Ingredient substitution: flavor-compound or nutrient based")
p.add_argument("ingredient", nargs="?")
p.add_argument("--mode", choices=["flavor", "nutrient"], default="flavor")
p.add_argument("--k", type=int, default=5)
p.add_argument("--same-category", action="store_true", help="only suggest substitutes from the same category")
p.add_argument("--min-confidence", choices=["Medium", "High"], help="hide lower-confidence suggestions")
p.add_argument("--include-variants", action="store_true", help="allow beef -> roasted_beef style suggestions")
p.add_argument("--include-generic", action="store_true", help="allow catch-all terms (meat, cheese, berry...)")
p.add_argument("--list", action="store_true")
g = p.add_argument_group("flavor mode")
g.add_argument("--weighting", choices=["jaccard", "idf"], default="jaccard")
g.add_argument("--no-ci", action="store_true", help="skip the bootstrap confidence intervals (faster)")
g = p.add_argument_group("nutrient mode")
g.add_argument("--catalog", choices=["all", "recipe"], default="all",
               help="all = 808 foods (344 recipe ingredients + 464 more FooDB foods); recipe = recipe ingredients only")
g.add_argument("--include-specialty", action="store_true",
               help="also suggest regional/traditional foods (e.g. Alaska Native foods such as bowhead whale)")
g.add_argument("--features", choices=["full", "macro"], default="full",
               help="full = 14 nutrients incl. minerals; macro = energy, protein, fat, carbs, fibre, sugar, alcohol")
g.add_argument("--lower", nargs="+", default=[], metavar="NUTRIENT", help="e.g. fat sugar sodium")
g.add_argument("--higher", nargs="+", default=[], metavar="NUTRIENT", help="e.g. protein fiber")
a = p.parse_args()

flavor = FlavorSubstitutionModel(weighting=a.weighting).fit()
model = flavor if a.mode == "flavor" else NutrientSubstitutionModel(features=a.features, catalog=a.catalog).fit()

if a.list or not a.ingredient:
    print(", ".join(sorted(model.ingredients["ingredient"])))
    if a.mode == "nutrient":
        print("\nnutrients usable with --lower/--higher:", ", ".join(sorted(GOAL_ALIASES)))
    raise SystemExit

q = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", a.ingredient.lower())).strip("_")
if q not in model:
    names = list(model.ingredients["ingredient"])
    close = [n for n in names if q in n] + difflib.get_close_matches(q, names, 5, 0.6)
    close = list(dict.fromkeys(close))
    why = "no flavor-compound data" if a.mode == "flavor" else "no usable nutrient profile in FooDB"
    raise SystemExit(f"'{q}' cannot be queried in {a.mode} mode ({why}, or not a recipe ingredient)."
                     + (f" Did you mean: {', '.join(close[:8])}?" if close else ""))

common = dict(exclude_variants=not a.include_variants, same_category=a.same_category,
              min_confidence=a.min_confidence, exclude_generic=not a.include_generic)
pd.set_option("display.width", 200)

if a.mode == "flavor":
    if not a.no_ci:
        flavor.bootstrap()
    r = flavor.recommend(q, a.k, **common)
    n = int(flavor.ingredients.set_index("ingredient").loc[q, "n_compounds"])
    print(f"\nFlavor substitutes for {q}  ({n} known flavor compounds, {a.weighting} similarity)\n")
    cols = ["rank", "substitute", "similarity"] + (["ci_low", "ci_high", "top5_stability"] if not a.no_ci else []) \
        + ["shared_compounds", "substitute_compounds", "confidence", "category"]
    print(r[cols].round(3).to_string(index=False))
    if n < 5:
        print(f"\nNote: {q} has only {n} known compound(s); these suggestions rest on very little chemical data.")
else:
    r = model.recommend(q, a.k, lower=a.lower, higher=a.higher, flavor_model=flavor,
                        exclude_specialty=not a.include_specialty, **common)
    prof = model.query_profile(q)
    goals = " ".join([f"lower {x}" for x in a.lower] + [f"higher {x}" for x in a.higher]) or "no goal"
    print(f"\nNutrient substitutes for {q}  ({goals}; {a.features} nutrient profile)")
    print(f"{q}: {prof['energy_kcal']:.0f} kcal, protein {prof['protein_g']:.1f} g, fat {prof['fat_g']:.1f} g, "
          f"carbs {prof['carbohydrate_g']:.1f} g, sugars {prof['sugars_g']:.1f} g, sodium {prof['sodium_mg']:.0f} mg  (per 100 g)\n")
    if r.empty:
        print("No ingredient satisfies these goals.")
    else:
        cols = ["rank", "substitute", "nutrient_similarity", "energy_kcal", "protein_g", "fat_g",
                "carbohydrate_g", "sugars_g", "sodium_mg", "confidence", "category", "flavor_similarity"]
        print(r[[c for c in cols if c in r]].round(2).to_string(index=False))
        if "flavor_similarity" in r:
            print("\nflavor_similarity = Jaccard flavor overlap with the query (taste check; NaN = no compound data)")
        if (r["origin"] == "foodb_auto").any():
            print("Foods outside the recipe dataset use automatic FooDB profiles (median of their basic products);"
                  " at most Medium confidence.")
