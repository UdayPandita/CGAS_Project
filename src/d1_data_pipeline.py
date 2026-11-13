"""
Deliverable 1 — Data Pipeline & Ingredient–Compound Matrix
==========================================================

Source data: Ahn, Ahnert, Bagrow & Barabási (2011),
"Flavor network and the principles of food pairing", Scientific Reports 1:196.

  data/raw/ingr_info.tsv        ingredient id, name, category          (1,530 rows)
  data/raw/comp_info.tsv        compound id, name, CAS number          (1,107 rows)
  data/raw/ingr_comp.tsv        ingredient id -> compound id edges     (36,781 rows)
  data/raw/srep00196-s2.csv     official shared-compound counts (used only to verify)
  data/raw/srep00196-s3.csv     56,498 recipes: cuisine, ingredient, ingredient, ...

Steps
  1. Load and clean the three flavor tables (types, whitespace, duplicates, dangling ids).
  2. Verify integrity against the paper's published shared-compound network (s2).
  3. Parse recipes into a tidy table, de-duplicate ingredients within a recipe,
     align recipe ingredient names to the flavor-table names.
  4. Build the binary ingredient x compound matrix  X[i, c] = 1 if ingredient i
     contains flavor compound c (this is the input to the Jaccard model in D2).
  5. Build the binary recipe x ingredient matrix (input to co-occurrence validation in D3).
  6. Save processed tables, matrices, summary statistics, and EDA figures.

Run:  python3 src/d1_data_pipeline.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
RES = ROOT / "results" / "d1"
OUT.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def norm(s: str) -> str:
    """Canonical ingredient/compound name: lowercase, trimmed, spaces -> underscores."""
    return "_".join(str(s).strip().lower().split())


# ---------------------------------------------------------------- 1. load + clean
def load_flavor_tables():
    ingr = pd.read_csv(RAW / "ingr_info.tsv", sep="\t", skiprows=1,
                       names=["ingredient_id", "ingredient", "category"])
    comp = pd.read_csv(RAW / "comp_info.tsv", sep="\t", skiprows=1,
                       names=["compound_id", "compound", "cas"])
    edges = pd.read_csv(RAW / "ingr_comp.tsv", sep="\t", skiprows=1,
                        names=["ingredient_id", "compound_id"])

    ingr["ingredient"] = ingr["ingredient"].map(norm)
    ingr["category"] = ingr["category"].str.strip()
    comp["compound"] = comp["compound"].map(norm)
    comp["cas"] = comp["cas"].astype(str).str.strip()

    log = {
        "raw_ingredients": len(ingr),
        "raw_compounds": len(comp),
        "raw_edges": len(edges),
        "duplicate_edges_removed": int(edges.duplicated().sum()),
        "duplicate_ingredient_names": int(ingr["ingredient"].duplicated().sum()),
        "duplicate_compound_names": int(comp["compound"].duplicated().sum()),
    }
    edges = edges.drop_duplicates()
    ok = edges["ingredient_id"].isin(ingr["ingredient_id"]) & edges["compound_id"].isin(comp["compound_id"])
    log["dangling_edges_removed"] = int((~ok).sum())
    edges = edges[ok]

    # ingredients / compounds that have no edges carry no chemical information
    no_comp = set(ingr["ingredient_id"]) - set(edges["ingredient_id"])
    log["ingredients_without_compounds"] = sorted(ingr.loc[ingr["ingredient_id"].isin(no_comp), "ingredient"])
    ingr = ingr[~ingr["ingredient_id"].isin(no_comp)].copy()
    unused = set(comp["compound_id"]) - set(edges["compound_id"])
    log["compounds_without_ingredients"] = len(unused)
    comp = comp[~comp["compound_id"].isin(unused)].copy()

    return ingr.reset_index(drop=True), comp.reset_index(drop=True), edges.reset_index(drop=True), log


# ---------------------------------------------------------------- 2. integrity check
def verify_against_paper(ingr, edges):
    """Recompute |C_a ∩ C_b| for every pair in the paper's s2 file and compare."""
    name = dict(zip(ingr["ingredient_id"], ingr["ingredient"]))
    sets = {name[i]: set(g) for i, g in edges.groupby("ingredient_id")["compound_id"]}
    s2 = pd.read_csv(RAW / "srep00196-s2.csv", comment="#", header=None, names=["a", "b", "shared"])
    s2["a"], s2["b"] = s2["a"].map(norm), s2["b"].map(norm)
    recomputed = [len(sets.get(a, set()) & sets.get(b, set())) for a, b in zip(s2["a"], s2["b"])]
    match = float(np.mean(np.array(recomputed) == s2["shared"].to_numpy()))
    return {"paper_pairs_checked": len(s2), "shared_compound_match_rate": match}


# ---------------------------------------------------------------- 3. recipes
def load_recipes(known: set[str]):
    rows, dup_within = [], 0
    with open(RAW / "srep00196-s3.csv") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p for p in line.strip().split(",") if p]
            cuisine, items = parts[0], [norm(p) for p in parts[1:]]
            uniq = list(dict.fromkeys(items))  # keep order, drop repeats
            dup_within += len(items) - len(uniq)
            rows.append((cuisine, uniq))
    rec = pd.DataFrame(rows, columns=["cuisine", "ingredients"])
    rec.insert(0, "recipe_id", range(len(rec)))
    rec["n_ingredients"] = rec["ingredients"].str.len()

    all_names = Counter(x for r in rec["ingredients"] for x in r)
    unmatched = sorted(set(all_names) - known)
    log = {
        "recipes": len(rec),
        "within_recipe_duplicates_removed": dup_within,
        "unique_recipe_ingredients": len(all_names),
        "recipe_ingredients_matched_to_compounds": len(set(all_names) & known),
        "recipe_ingredients_unmatched": unmatched,
    }
    # drop unmatched ingredients (none expected) and recipes left with <2 ingredients
    if unmatched:
        rec["ingredients"] = rec["ingredients"].map(lambda r: [x for x in r if x in known])
        rec["n_ingredients"] = rec["ingredients"].str.len()
    before = len(rec)
    rec = rec[rec["n_ingredients"] >= 2].reset_index(drop=True)
    log["recipes_dropped_lt2_ingredients"] = before - len(rec)
    return rec, Counter(x for r in rec["ingredients"] for x in r), log


# ---------------------------------------------------------------- 4/5. matrices
def build_matrices(ingr, comp, edges, rec):
    ingr = ingr.sort_values("ingredient").reset_index(drop=True)
    comp = comp.sort_values("compound").reset_index(drop=True)
    row = {iid: k for k, iid in enumerate(ingr["ingredient_id"])}
    col = {cid: k for k, cid in enumerate(comp["compound_id"])}
    X = sparse.csr_matrix(
        (np.ones(len(edges), dtype=np.uint8),
         (edges["ingredient_id"].map(row), edges["compound_id"].map(col))),
        shape=(len(ingr), len(comp)))

    idx = {n: k for k, n in enumerate(ingr["ingredient"])}
    r_idx = np.repeat(np.arange(len(rec)), rec["n_ingredients"].to_numpy())
    c_idx = np.fromiter((idx[x] for r in rec["ingredients"] for x in r), dtype=np.int64)
    R = sparse.csr_matrix((np.ones(len(c_idx), dtype=np.uint8), (r_idx, c_idx)),
                          shape=(len(rec), len(ingr)))
    return ingr, comp, X, R


# ---------------------------------------------------------------- 6. figures
def make_figures(ingr, comp, rec):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    blue, grey = "#2f6db3", "#9aa3ad"

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    a = ax[0, 0]
    a.hist(ingr["n_compounds"], bins=np.logspace(0, np.log10(ingr["n_compounds"].max() + 1), 40),
           color=grey, label="all ingredients")
    a.hist(ingr.loc[ingr["in_recipes"], "n_compounds"],
           bins=np.logspace(0, np.log10(ingr["n_compounds"].max() + 1), 40), color=blue,
           label="used in recipes")
    a.set_xscale("log"); a.set_xlabel("flavor compounds per ingredient"); a.set_ylabel("ingredients")
    a.set_title("A. Compound profile size"); a.legend(frameon=False)

    a = ax[0, 1]
    a.hist(comp["n_ingredients"], bins=np.logspace(0, np.log10(comp["n_ingredients"].max() + 1), 40),
           color=blue)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("ingredients containing compound"); a.set_ylabel("compounds")
    a.set_title("B. Compound prevalence")

    a = ax[0, 2]
    cats = ingr.groupby("category").agg(all=("ingredient", "size"), used=("in_recipes", "sum"))
    cats = cats.sort_values("all")
    a.barh(cats.index, cats["all"], color=grey, label="all")
    a.barh(cats.index, cats["used"], color=blue, label="used in recipes")
    a.set_xlabel("ingredients"); a.set_title("C. Ingredient categories"); a.legend(frameon=False)

    a = ax[1, 0]
    cu = rec["cuisine"].value_counts().sort_values()
    a.barh(cu.index, cu.values, color=blue); a.set_xscale("log")
    a.set_xlabel("recipes (log)"); a.set_title("D. Recipes per cuisine")

    a = ax[1, 1]
    vc = rec["n_ingredients"].value_counts().sort_index()
    a.bar(vc.index, vc.values, color=blue, width=0.85)
    a.axvline(rec["n_ingredients"].mean(), color="k", lw=1, ls="--",
              label=f"mean = {rec['n_ingredients'].mean():.1f}")
    a.set_xlabel("ingredients per recipe"); a.set_ylabel("recipes"); a.set_title("E. Recipe size")
    a.legend(frameon=False)

    a = ax[1, 2]
    top = ingr.nlargest(20, "recipe_freq").iloc[::-1]
    a.barh(top["ingredient"], top["recipe_freq"] / len(rec) * 100, color=blue)
    a.set_xlabel("% of recipes"); a.set_title("F. Most frequent recipe ingredients")

    fig.tight_layout()
    fig.savefig(RES / "d1_eda_overview.png", bbox_inches="tight")
    plt.close(fig)


def make_matrix_figure(ingr, X):
    """Spy-plot of the ingredient x compound matrix for recipe ingredients, sorted by category."""
    used = ingr[ingr["in_recipes"]].sort_values(["category", "n_compounds"], ascending=[True, False])
    Xu = X[used.index.to_numpy()]
    col_order = np.argsort(-np.asarray(Xu.sum(0)).ravel())
    Xu = Xu[:, col_order]
    Xu = Xu[:, np.asarray(Xu.sum(0)).ravel() > 0]
    fig, a = plt.subplots(figsize=(11, 7))
    a.spy(Xu, markersize=0.4, color="#2f6db3", aspect="auto")
    b = used["category"].to_numpy()
    edges = [0] + [k for k in range(1, len(b)) if b[k] != b[k - 1]] + [len(b)]
    for s, e in zip(edges[:-1], edges[1:]):
        a.axhline(s - 0.5, color="#ccc", lw=0.5)
        a.text(-8, (s + e) / 2, b[s], ha="right", va="center", fontsize=7)
    a.set_yticks([]); a.set_xlabel("flavor compounds (sorted by prevalence)")
    a.set_title(f"Ingredient–compound matrix: {Xu.shape[0]} recipe ingredients × {Xu.shape[1]} compounds "
                f"(density {Xu.nnz / (Xu.shape[0] * Xu.shape[1]):.1%})", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "d1_matrix_spy.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ingr, comp, edges, log_clean = load_flavor_tables()
    log_verify = verify_against_paper(ingr, edges)
    rec, freq, log_rec = load_recipes(set(ingr["ingredient"]))

    ingr, comp, X, R = build_matrices(ingr, comp, edges, rec)
    ingr["n_compounds"] = np.asarray(X.sum(1)).ravel()
    ingr["recipe_freq"] = np.asarray(R.sum(0)).ravel()
    ingr["in_recipes"] = ingr["recipe_freq"] > 0
    comp["n_ingredients"] = np.asarray(X.sum(0)).ravel()
    used = ingr["in_recipes"].to_numpy()
    comp["n_recipe_ingredients"] = np.asarray(X[used].sum(0)).ravel()

    # ---- save
    ingr.to_csv(OUT / "ingredients.csv", index_label="row")
    comp.to_csv(OUT / "compounds.csv", index_label="col")
    out_rec = rec.assign(ingredients=rec["ingredients"].str.join(";"))
    out_rec.to_csv(OUT / "recipes.csv", index=False)
    sparse.save_npz(OUT / "ingredient_compound_matrix.npz", X)
    sparse.save_npz(OUT / "recipe_ingredient_matrix.npz", R)
    # human-readable version of the matrix for the 381 recipe ingredients
    Xu = X[used][:, comp["n_recipe_ingredients"].to_numpy() > 0]
    pd.DataFrame(Xu.toarray(), index=ingr.loc[used, "ingredient"],
                 columns=comp.loc[comp["n_recipe_ingredients"] > 0, "compound"]
                 ).to_csv(OUT / "ingredient_compound_matrix_recipe_ingredients.csv")

    # ---- consistency checks between tables and matrices
    assert X.shape == (len(ingr), len(comp)), "X shape != ingredients x compounds"
    assert X.nnz == len(edges), "matrix nonzeros != ingredient-compound links"
    assert R.shape == (len(rec), len(ingr)), "R shape != recipes x ingredients"
    assert R.nnz == rec["n_ingredients"].sum(), "R nonzeros != recipe ingredient mentions"
    assert Xu.shape == (int(used.sum()), int((comp["n_recipe_ingredients"] > 0).sum()))
    assert (np.asarray(Xu.sum(1)).ravel() == ingr.loc[used, "n_compounds"].to_numpy()).all(), \
        "recipe-ingredient CSV lost compounds"
    print(f"consistency checks passed: full matrix {X.shape}, recipe-ingredient CSV {Xu.shape}, "
          f"recipe matrix {R.shape}")

    # ---- summary
    nu = ingr[ingr["in_recipes"]]
    summary = {
        "cleaning": log_clean,
        "verification": log_verify,
        "recipes": log_rec,
        "matrix_all": {"ingredients": X.shape[0], "compounds": X.shape[1], "nonzeros": int(X.nnz),
                        "density": X.nnz / (X.shape[0] * X.shape[1])},
        "matrix_recipe_ingredients": {"ingredients": int(Xu.shape[0]), "compounds": int(Xu.shape[1]),
                                       "nonzeros": int(Xu.nnz),
                                       "density": Xu.nnz / (Xu.shape[0] * Xu.shape[1])},
        "compounds_per_ingredient_all": ingr["n_compounds"].describe().round(2).to_dict(),
        "compounds_per_ingredient_recipe": nu["n_compounds"].describe().round(2).to_dict(),
        "ingredients_per_compound": comp["n_ingredients"].describe().round(2).to_dict(),
        "recipe_size": rec["n_ingredients"].describe().round(2).to_dict(),
        "recipes_per_cuisine": rec["cuisine"].value_counts().to_dict(),
        "top_compound_rich_recipe_ingredients": nu.nlargest(10, "n_compounds")[["ingredient", "n_compounds"]].values.tolist(),
        "sparse_recipe_ingredients_le3_compounds": int((nu["n_compounds"] <= 3).sum()),
        "most_prevalent_compounds": comp.nlargest(10, "n_ingredients")[["compound", "n_ingredients"]].values.tolist(),
    }
    (RES / "d1_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    make_figures(ingr, comp, rec)
    make_matrix_figure(ingr, X)

    print(json.dumps({k: summary[k] for k in ["cleaning", "verification", "matrix_all",
                                              "matrix_recipe_ingredients"]}, indent=2, default=str))
    print("recipes:", {k: v for k, v in log_rec.items() if k != "recipe_ingredients_unmatched"})
    print("compounds/ingredient (recipe ingr):", summary["compounds_per_ingredient_recipe"])
    print("sparse recipe ingredients (<=3 compounds):", summary["sparse_recipe_ingredients_le3_compounds"])
    print("top compound-rich:", summary["top_compound_rich_recipe_ingredients"])
    print("most prevalent compounds:", summary["most_prevalent_compounds"])
    print("recipe size:", summary["recipe_size"])


if __name__ == "__main__":
    main()
