"""
Substitution service: loads the D2 models once and answers queries for the web app.

Everything the service needs is small and already produced by the pipeline:
  data/processed/*            flavor matrix, nutrient tables, 808-food catalog
  results/d3/ground_truth_pairs.csv   real substitutions (Recipe1MSubs, mapped in D3)
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from substitution_model import FlavorSubstitutionModel, NutrientSubstitutionModel  # noqa: E402

PROC = ROOT / "data" / "processed"
NUTRIENT_COLS = ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugars_g",
                 "alcohol_g", "cholesterol_mg", "sodium_mg", "potassium_mg", "calcium_mg",
                 "iron_mg", "phosphorus_mg", "zinc_mg"]


def _clean(v):
    """JSON-safe scalar: NaN/inf -> None, numpy -> python."""
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else round(v, 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def normalise(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


class SubstitutionService:
    def __init__(self, bootstrap: int = 200):
        self.flavor = {
            "jaccard": FlavorSubstitutionModel(weighting="jaccard").fit().bootstrap(B=bootstrap),
            "idf": FlavorSubstitutionModel(weighting="idf").fit().bootstrap(B=bootstrap),
        }
        self.nutrient = {(c, f): NutrientSubstitutionModel(features=f, catalog=c).fit()
                         for c in ("all", "recipe") for f in ("full", "macro")}

        # compound names for "shared compounds" explanations
        self._X = sparse.load_npz(PROC / "ingredient_compound_matrix.npz").tocsr()
        self._compounds = pd.read_csv(PROC / "compounds.csv").sort_values("col")
        ing = pd.read_csv(PROC / "ingredients.csv")
        self._ing = ing.set_index("ingredient")

        # real substitutions reported by cooks (Recipe1MSubs, mapped onto our names in D3)
        gt_path = ROOT / "results" / "d3" / "ground_truth_pairs.csv"
        gt = pd.read_csv(gt_path) if gt_path.exists() else pd.DataFrame(columns=["query", "sub", "count"])
        self._gt = {(a, b): int(c) for a, b, c in zip(gt["query"], gt["sub"], gt["count"])}
        self._gt_by_query = {q: g.sort_values("count", ascending=False)[["sub", "count"]].values.tolist()
                             for q, g in gt.groupby("query")}

        cat = self.nutrient[("all", "full")].ingredients
        self._display = dict(zip(cat["ingredient"], cat["display_name"]))
        self._index = self._build_index()

    # ------------------------------------------------------------------ lookup
    def _build_index(self) -> list[dict]:
        fl = self.flavor["jaccard"]
        nall = self.nutrient[("all", "full")]
        names = sorted(set(fl.ingredients["ingredient"]) | set(nall.ingredients["ingredient"]))
        out = []
        ncat = nall.ingredients.set_index("ingredient")
        for n in names:
            in_f, in_n = n in fl, n in nall
            out.append({
                "name": n,
                "label": self._display.get(n, n).replace("_", " "),
                "flavor": in_f,
                "nutrient": in_n,
                "recipe": bool(n in self._ing.index and self._ing.loc[n, "in_recipes"]),
                "group": (ncat.loc[n, "food_group"] if in_n else self._ing.loc[n, "category"]) if (in_n or n in self._ing.index) else None,
                "compounds": int(self._ing.loc[n, "n_compounds"]) if in_f else None,
            })
        return out

    def search(self, q: str, limit: int = 12) -> list[dict]:
        q = normalise(q)
        if not q:
            return []
        starts = [r for r in self._index if r["name"].startswith(q)]
        contains = [r for r in self._index if q in r["name"] and not r["name"].startswith(q)]
        hits = starts + contains
        hits.sort(key=lambda r: (not r["name"].startswith(q), not r["recipe"], len(r["name"])))
        return [{k: _clean(v) for k, v in r.items()} for r in hits[:limit]]

    def resolve(self, name: str) -> str | None:
        n = normalise(name)
        return n if any(r["name"] == n for r in self._index) else None

    # ------------------------------------------------------------------ ingredient card
    def ingredient(self, name: str) -> dict:
        n = self.resolve(name)
        if n is None:
            raise KeyError(name)
        row = next(r for r in self._index if r["name"] == n)
        out = {k: _clean(v) for k, v in row.items()}
        nall = self.nutrient[("all", "full")]
        if n in nall:
            prof = nall.ingredients.set_index("ingredient").loc[n]
            out["nutrients"] = {c: _clean(prof[c]) for c in NUTRIENT_COLS}
            out["nutrient_source"] = {"origin": prof["origin"], "match_type": prof["match_type"],
                                      "description": prof["description"], "n_imputed": _clean(prof["n_imputed"])}
        out["real_substitutes"] = [{"name": s, "count": c} for s, c in self._gt_by_query.get(n, [])[:8]]
        return out

    # ------------------------------------------------------------------ recommendations
    def recommend(self, name: str, mode: str = "flavor", k: int = 8, weighting: str = "jaccard",
                  catalog: str = "all", features: str = "full", lower=(), higher=(),
                  same_category: bool = False, min_confidence: str | None = None) -> dict:
        n = self.resolve(name)
        if n is None:
            raise KeyError(name)
        k = max(1, min(int(k), 25))
        fl = self.flavor.get(weighting, self.flavor["jaccard"])
        common = dict(same_category=same_category, min_confidence=min_confidence or None)

        if mode == "flavor":
            if n not in fl:
                raise LookupError(f"'{n}' has no flavor-compound data. Try nutrient mode.")
            r = fl.recommend(n, k, **common)
            cols = ["rank", "substitute", "similarity", "ci_low", "ci_high", "top5_stability",
                    "shared_compounds", "query_compounds", "substitute_compounds", "confidence",
                    "category", "same_category", "name_nested"]
            r = r[cols]
            note = None
            qsize = int(self._ing.loc[n, "n_compounds"])
            if qsize < 5:
                note = (f"{n.replace('_', ' ')} has only {qsize} known flavor compound"
                        f"{'s' if qsize != 1 else ''}, so these suggestions rest on very little chemical data.")
        else:
            nm = self.nutrient.get((catalog, features), self.nutrient[("all", "full")])
            if n not in nm:
                raise LookupError(f"'{n}' is not in the {catalog} nutrient catalog.")
            r = nm.recommend(n, k, lower=list(lower), higher=list(higher), flavor_model=fl, **common)
            keep = ["rank", "substitute", "nutrient_similarity", "confidence", "category", "origin",
                    "match_type", "identical_record", "same_category", "flavor_similarity"]
            r = r[[c for c in keep if c in r]]
            prof = nm.ingredients.set_index("ingredient")
            for c in NUTRIENT_COLS:
                r[c] = prof.loc[r["substitute"], c].to_numpy() if len(r) else []
            note = None

        r = r.copy()
        r["label"] = [self._display.get(s, s).replace("_", " ") for s in r["substitute"]]
        r["cooks_used"] = [self._gt.get((n, s), 0) for s in r["substitute"]]
        return {"query": n, "mode": mode, "note": note, "results": _records(r)}

    def shared_compounds(self, a: str, b: str, limit: int = 40) -> dict:
        a, b = self.resolve(a), self.resolve(b)
        if a is None or b is None or a not in self._ing.index or b not in self._ing.index:
            raise KeyError("unknown ingredient")
        ra, rb = int(self._ing.loc[a, "row"]), int(self._ing.loc[b, "row"])
        ca, cb = set(self._X[ra].indices), set(self._X[rb].indices)
        shared = sorted(ca & cb)
        comp = self._compounds.set_index("col")
        # rare compounds first: they say more about the pair than ubiquitous ones
        shared.sort(key=lambda c: comp.loc[c, "n_ingredients"])
        return {"a": a, "b": b, "n_a": len(ca), "n_b": len(cb), "n_shared": len(shared),
                "compounds": [{"name": comp.loc[c, "compound"].replace("_", " "),
                               "found_in": int(comp.loc[c, "n_ingredients"])} for c in shared[:limit]]}
