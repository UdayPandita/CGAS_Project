"""
Ingredient substitution models (Deliverable 2).

Two independent ways to find a substitute, chosen by the user:

1. FLAVOR mode — FlavorSubstitutionModel
   Each ingredient i is its set of flavor compounds C_i (row of the D1 ingredient x compound
   matrix). Candidates are ranked by
        Jaccard            J(a, b)  = |C_a ∩ C_b| / |C_a ∪ C_b|
        IDF-weighted       Jw(a, b) = Σ_{c∈C_a∩C_b} w_c / Σ_{c∈C_a∪C_b} w_c,   w_c = log(N / df_c)
   Uncertainty: every ingredient is scored (no hard cutoff). Each recommendation carries
     * a 95% bootstrap confidence interval for the similarity (compounds resampled with
       replacement, B = 200),
     * top-k stability: how often the substitute stays in the top-k across bootstrap resamples,
     * a confidence label from profile sizes: High (both >= 20 compounds),
       Medium (both >= 5), Low (otherwise).

2. NUTRIENT mode — NutrientSubstitutionModel
   Each ingredient is its FooDB/USDA nutrient vector per 100 g (D1 part B). Values are
   log1p-transformed and z-scored, and candidates are ranked by the root-mean-square
   z-difference d(a, b); nutrient similarity = 1 / (1 + d). Optional goals keep only
   candidates that are lower/higher in chosen nutrients (e.g. lower fat, higher protein).
   Confidence label from data quality: High (both matched directly, nothing imputed),
   Medium (proxy match or <= 2 imputed values), Low (otherwise).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# tokens that describe a preparation/form rather than a different ingredient
_FORM_WORDS = {"raw", "roasted", "cooked", "boiled", "fried", "dried", "smoked", "fresh",
               "whole", "ground", "toasted", "baked", "juice", "oil", "seed", "leaf",
               "powder", "peel", "extract", "grain", "flour"}

NUTRIENT_FEATURES = ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugars_g",
                     "alcohol_g", "cholesterol_mg", "sodium_mg", "potassium_mg", "calcium_mg",
                     "iron_mg", "phosphorus_mg", "zinc_mg"]
MACRO_FEATURES = ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugars_g", "alcohol_g"]
GOAL_ALIASES = {"energy": "energy_kcal", "calories": "energy_kcal", "kcal": "energy_kcal",
                "protein": "protein_g", "fat": "fat_g", "carbs": "carbohydrate_g",
                "carbohydrate": "carbohydrate_g", "fiber": "fiber_g", "fibre": "fiber_g",
                "sugar": "sugars_g", "sugars": "sugars_g", "alcohol": "alcohol_g",
                "cholesterol": "cholesterol_mg", "sodium": "sodium_mg", "salt": "sodium_mg",
                "potassium": "potassium_mg", "calcium": "calcium_mg", "iron": "iron_mg",
                "phosphorus": "phosphorus_mg", "zinc": "zinc_mg"}


# catch-all recipe terms: they can be queried, but are not useful as suggested substitutes
GENERIC = {"meat", "fish", "cheese", "nut", "seed", "berry", "citrus", "cereal", "shellfish",
           "vegetable", "fruit", "flower", "leaf", "root", "condiment", "brassica", "roasted_meat",
           "roasted_nut", "smoked_fish", "wood", "smoke"}


def _base_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[_\s]+", name) if t and t not in _FORM_WORDS}


def is_variant(a: str, b: str) -> bool:
    """True when two names are the *same* ingredient in a different form, i.e. their
    names are identical once preparation/form words are removed
    (beef / raw_beef / roasted_beef, potato / baked_potato, orange / orange_juice).
    Such pairs are trivial 'substitutes' and are excluded from recommendations."""
    ta, tb = _base_tokens(a), _base_tokens(b)
    return bool(ta) and ta == tb


def is_name_nested(a: str, b: str) -> bool:
    """True when one name is a more specific kind of the other
    (cheese / parmesan_cheese, pepper / black_pepper, wine / white_wine).
    These are legitimate but 'easy' substitutes; they are kept and flagged."""
    ta, tb = _base_tokens(a), _base_tokens(b)
    return bool(ta) and bool(tb) and ta != tb and (ta <= tb or tb <= ta)


def _rank(name: str, cand: pd.DataFrame, score: str, k: int, exclude_variants: bool,
          exclude_generic: bool = True) -> pd.DataFrame:
    """Shared ranking: drop generic catch-all terms and forms of the query, collapse forms
    among candidates, take top k."""
    cand = cand.copy()
    if exclude_generic:
        cand = cand[~cand["substitute"].isin(GENERIC)]
    cand["variant"] = [is_variant(name, s) for s in cand["substitute"]]
    cand["name_nested"] = [is_name_nested(name, s) for s in cand["substitute"]]
    if exclude_variants:
        cand = cand[~cand["variant"]]
    cand = cand.sort_values([score, "substitute"], ascending=[False, True])
    if exclude_variants:
        kept, rows = [], []
        for r, s in enumerate(cand["substitute"]):
            if not any(is_variant(s, t) for t in kept):
                kept.append(s)
                rows.append(r)
            if len(kept) == k:
                break
        cand = cand.iloc[rows]
    cand = cand.head(k)
    cand.insert(0, "rank", range(1, len(cand) + 1))
    cand.insert(0, "ingredient", name)
    return cand.reset_index(drop=True)


def _confidence_from_sizes(na: int, nb: np.ndarray) -> np.ndarray:
    lo = np.minimum(na, nb)
    return np.where(lo >= 20, "High", np.where(lo >= 5, "Medium", "Low"))


# ====================================================================== FLAVOR
@dataclass
class FlavorSubstitutionModel:
    min_compounds: int = 1              # 1 = score everything; uncertainty is reported instead
    weighting: str = "jaccard"          # "jaccard" or "idf"
    pool: str = "recipe"                # "recipe" (381 ingredients) or "all" (1,525)
    ingredients: pd.DataFrame = field(init=False, repr=False)
    S: np.ndarray = field(init=False, repr=False)            # similarity matrix
    shared: np.ndarray = field(init=False, repr=False)       # |C_a ∩ C_b|
    ci_lo: np.ndarray | None = field(default=None, init=False, repr=False)
    ci_hi: np.ndarray | None = field(default=None, init=False, repr=False)
    topk_stability: np.ndarray | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(self, X: sparse.csr_matrix | None = None, ingredients: pd.DataFrame | None = None):
        if X is None:
            X = sparse.load_npz(PROC / "ingredient_compound_matrix.npz")
        if ingredients is None:
            ingredients = pd.read_csv(PROC / "ingredients.csv")

        # IDF over the full 1,525-ingredient table: compound rarity reflects all of chemistry
        df = np.asarray(X.sum(0)).ravel()
        self._idf = np.log(X.shape[0] / np.maximum(df, 1))

        keep = ingredients["n_compounds"] >= self.min_compounds
        if self.pool == "recipe":
            keep &= ingredients["in_recipes"]
        ing = ingredients[keep].reset_index(drop=True)
        Xk = X[ing["row"].to_numpy()].astype(np.float64).tocsr()
        Xk = Xk[:, np.asarray(Xk.sum(0)).ravel() > 0]          # drop compounds absent from pool
        self._X = Xk
        self._idf_k = self._idf[np.asarray(X[ing["row"].to_numpy()].sum(0)).ravel() > 0]

        w = np.ones(Xk.shape[1]) if self.weighting == "jaccard" else self._idf_k
        if self.weighting not in ("jaccard", "idf"):
            raise ValueError(f"unknown weighting {self.weighting!r}")
        S = self._similarity(w)
        np.fill_diagonal(S, np.nan)                              # not its own substitute

        self.ingredients = ing
        self.S = S
        self.shared = (Xk @ Xk.T).toarray().astype(int)
        self._idx = {n: k for k, n in enumerate(ing["ingredient"])}
        self._cats = ing["category"].to_numpy()
        self._names = ing["ingredient"].to_numpy()
        self._sizes = ing["n_compounds"].to_numpy()
        return self

    def _similarity(self, w: np.ndarray) -> np.ndarray:
        """Weighted Jaccard for compound weights w (w = 1 -> plain Jaccard).
        Pairs with zero union weight get similarity 0 (no evidence)."""
        Xw = self._X.multiply(w).tocsr()
        inter = (Xw @ self._X.T).toarray()
        size = np.asarray(Xw.sum(1)).ravel()
        union = size[:, None] + size[None, :] - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    # ------------------------------------------------------------------ uncertainty
    def bootstrap(self, B: int = 200, k: int = 5, seed: int = 0, alpha: float = 0.05):
        """Resample compounds (columns) with replacement B times and recompute similarity.
        Stores 95% percentile CIs and the probability that each candidate is in the
        query's top-k (form variants of the query excluded)."""
        rng = np.random.default_rng(seed)
        n, m = len(self._names), self._X.shape[1]
        base_w = np.ones(m) if self.weighting == "jaccard" else self._idf_k
        V = np.array([[is_variant(a, b) for b in self._names] for a in self._names])
        np.fill_diagonal(V, True)
        draws = np.empty((B, n, n), dtype=np.float32)
        inclusion = np.zeros((n, n), dtype=np.int32)
        for b in range(B):
            counts = rng.multinomial(m, np.full(m, 1 / m))
            Sb = self._similarity(base_w * counts)
            draws[b] = Sb
            Sb = np.where(V, -1.0, Sb)
            top = np.argpartition(-Sb, k, axis=1)[:, :k]
            np.add.at(inclusion, (np.repeat(np.arange(n), k), top.ravel()), 1)
        self.ci_lo = np.percentile(draws, 100 * alpha / 2, axis=0)
        self.ci_hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
        self.topk_stability = inclusion / B
        self._boot_k = k
        return self

    # ------------------------------------------------------------------ query
    def __contains__(self, name: str) -> bool:
        return name in self._idx

    def similarity(self, a: str, b: str) -> float:
        return float(self.S[self._idx[a], self._idx[b]])

    def recommend(self, name: str, k: int = 10, exclude_variants: bool = True,
                  same_category: bool = False, min_confidence: str | None = None,
                  exclude_generic: bool = True) -> pd.DataFrame:
        """Top-k substitutes for `name`, ranked by similarity (ties broken alphabetically).

        exclude_variants: drop other forms of the query itself (beef -> roasted_beef) and
        collapse repeated forms among the candidates, so each slot is a distinct ingredient.
        min_confidence: 'Medium' or 'High' hides lower-confidence candidates."""
        if name not in self._idx:
            raise KeyError(f"{name!r} is not a recipe ingredient with flavor-compound data")
        i = self._idx[name]
        cand = pd.DataFrame({
            "substitute": self._names,
            "similarity": self.S[i],
            "shared_compounds": self.shared[i],
            "query_compounds": self._sizes[i],
            "substitute_compounds": self._sizes,
            "category": self._cats,
            "confidence": _confidence_from_sizes(self._sizes[i], self._sizes),
        })
        if self.ci_lo is not None:
            cand["ci_low"] = self.ci_lo[i]
            cand["ci_high"] = self.ci_hi[i]
            cand[f"top{self._boot_k}_stability"] = self.topk_stability[i]
        cand = cand.drop(index=i)
        cand["same_category"] = cand["category"] == self._cats[i]
        if same_category:
            cand = cand[cand["same_category"]]
        if min_confidence:
            allowed = {"High": ["High"], "Medium": ["High", "Medium"]}[min_confidence]
            cand = cand[cand["confidence"].isin(allowed)]
        return _rank(name, cand, "similarity", k, exclude_variants, exclude_generic)

    def recommend_all(self, k: int = 10, **kw) -> pd.DataFrame:
        return pd.concat([self.recommend(n, k, **kw) for n in self._names], ignore_index=True)

    def similarity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.S, index=self._names, columns=self._names)


# ====================================================================== NUTRIENT
@dataclass
class NutrientSubstitutionModel:
    features: str = "full"              # "full" (14 nutrients) or "macro" (7 macronutrients)
    catalog: str = "recipe"             # "recipe": 344 curated recipe ingredients
                                        # "all":    + every other FooDB food with a nutrient profile (808)
    ingredients: pd.DataFrame = field(init=False, repr=False)
    D: np.ndarray = field(init=False, repr=False)            # RMS z-distance matrix
    S: np.ndarray = field(init=False, repr=False)            # 1 / (1 + D)

    def fit(self, nutrients: pd.DataFrame | None = None):
        if nutrients is None:
            f = "ingredient_nutrients.csv" if self.catalog == "recipe" else "food_nutrient_catalog.csv"
            nutrients = pd.read_csv(PROC / f)
        ing = nutrients[nutrients["match_type"] != "none"].reset_index(drop=True)
        if "origin" not in ing:
            ing["origin"] = "recipe_curated"
        if "specialty" not in ing:
            ing["specialty"] = False
        # category used for --same-category: recipe categories for the recipe catalog,
        # FooDB food groups for the extended catalog (auto-added foods have no recipe category)
        self._catcol = "category" if self.catalog == "recipe" else "food_group"
        cols = NUTRIENT_FEATURES if self.features == "full" else MACRO_FEATURES
        Z = np.log1p(ing[cols].to_numpy(dtype=float))
        Z = (Z - Z.mean(0)) / Z.std(0)
        diff = Z[:, None, :] - Z[None, :, :]
        D = np.sqrt((diff ** 2).mean(-1))
        S = 1 / (1 + D)
        np.fill_diagonal(S, np.nan)
        self.ingredients, self.D, self.S, self._cols = ing, D, S, cols
        self._idx = {n: k for k, n in enumerate(ing["ingredient"])}
        self._names = ing["ingredient"].to_numpy()
        self._vals = ing[NUTRIENT_FEATURES]
        # High: hand-matched directly, nothing imputed. Medium: proxy match, <= 2 imputed, or an
        # automatically aggregated FooDB food with nothing imputed. Low: otherwise.
        q = ing["match_type"].eq("direct") & ing["n_imputed"].eq(0)
        m = (ing["match_type"].isin(["proxy", "direct"]) & ing["n_imputed"].le(2)) | \
            (ing["match_type"].eq("auto") & ing["n_imputed"].eq(0))
        self._quality = np.where(q, 2, np.where(m, 1, 0))
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._idx

    def recommend(self, name: str, k: int = 10, lower: list[str] | None = None,
                  higher: list[str] | None = None, margin: float = 0.10,
                  exclude_variants: bool = True, same_category: bool = False,
                  min_confidence: str | None = None, flavor_model: FlavorSubstitutionModel | None = None,
                  exclude_generic: bool = True, exclude_specialty: bool = True) -> pd.DataFrame:
        """Top-k nutritionally similar substitutes.

        exclude_specialty: do not suggest regional/traditional foods known only from USDA's
        Alaska Native / Native American records (e.g. bowhead whale); they can still be queried.

        lower / higher: nutrient names (e.g. ['fat'], ['protein']); keeps only candidates at
        least `margin` (10%) lower / higher than the query in those nutrients."""
        if name not in self._idx:
            raise KeyError(f"{name!r} is not in the {self.catalog!r} nutrient catalog")
        i = self._idx[name]
        q = self._vals.iloc[i]
        cand = pd.DataFrame({
            "substitute": self._names,
            "nutrient_similarity": self.S[i],
            "distance": self.D[i],
            "category": self.ingredients[self._catcol].to_numpy(),
            "origin": self.ingredients["origin"].to_numpy(),
            "match_type": self.ingredients["match_type"].to_numpy(),
            "description": self.ingredients["description"].to_numpy(),
            "confidence": np.array(["Low", "Medium", "High"])[np.minimum(self._quality[i], self._quality)],
        })
        for f in ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g", "sugars_g", "sodium_mg"]:
            cand[f] = self._vals[f].to_numpy()
        cand["identical_record"] = cand["description"] == self.ingredients.loc[i, "description"]
        cand = cand.drop(index=i)
        if exclude_specialty:
            cand = cand[~self.ingredients["specialty"].to_numpy(dtype=bool)[cand.index]]
        cand["same_category"] = cand["category"] == self.ingredients.loc[i, self._catcol]
        for goal, sign in [(lower or [], -1), (higher or [], +1)]:
            for g in goal:
                col = GOAL_ALIASES.get(g.lower(), g)
                if col not in NUTRIENT_FEATURES:
                    raise ValueError(f"unknown nutrient {g!r}; options: {sorted(GOAL_ALIASES)}")
                v = self._vals[col].to_numpy()[cand.index]
                cand = cand[(v <= q[col] * (1 - margin)) if sign < 0 else (v >= q[col] * (1 + margin))
                            ] if len(cand) else cand
        if same_category:
            cand = cand[cand["same_category"]]
        if min_confidence:
            allowed = {"High": ["High"], "Medium": ["High", "Medium"]}[min_confidence]
            cand = cand[cand["confidence"].isin(allowed)]
        if flavor_model is not None and name in flavor_model:
            cand["flavor_similarity"] = [flavor_model.similarity(name, s) if s in flavor_model else np.nan
                                         for s in cand["substitute"]]
        return _rank(name, cand, "nutrient_similarity", k, exclude_variants, exclude_generic)

    def query_profile(self, name: str) -> pd.Series:
        return self._vals.iloc[self._idx[name]]

    def recommend_all(self, k: int = 10, **kw) -> pd.DataFrame:
        return pd.concat([self.recommend(n, k, **kw) for n in self._names], ignore_index=True)

    def similarity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.S, index=self._names, columns=self._names)
