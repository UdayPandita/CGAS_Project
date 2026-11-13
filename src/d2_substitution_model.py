"""
Deliverable 2 — Flavor- and Nutrient-based Substitution Models
==============================================================

Fits both models from src/substitution_model.py and writes

  results/d2/recommendations_flavor_jaccard.csv   top-10 per ingredient, with 95% CI, top-5 stability, confidence
  results/d2/recommendations_flavor_idf.csv       same for IDF-weighted Jaccard
  results/d2/recommendations_nutrient.csv         top-10 nutrient substitutes per recipe ingredient, with confidence
  results/d2/recommendations_nutrient_all_foods.csv  same over the extended 808-food FooDB catalog
  results/d2/goal_examples.csv                    nutrient mode with goals (lower fat, higher protein, ...)
  results/d2/similarity_flavor_jaccard.csv / similarity_nutrient.csv
  results/d2/d2_summary.json                      every number used in the report
  results/d2/*.png                                figures

Run:  python3 src/d2_substitution_model.py      (~30 s)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import sparse

from substitution_model import (GENERIC, PROC, ROOT, FlavorSubstitutionModel,
                                NutrientSubstitutionModel)

RES = ROOT / "results" / "d2"
RES.mkdir(parents=True, exist_ok=True)
for old in ["recommendations_jaccard.csv", "recommendations_idf.csv", "similarity_jaccard.csv",
            "similarity_idf.csv", "d2_threshold.png", "threshold_stability.csv"]:
    (RES / old).unlink(missing_ok=True)          # outputs of the earlier cutoff-based version

TOP_K = 10
CASES = ["butter", "parmesan_cheese", "lemon", "basil", "chicken", "coffee"]
GOALS = [("butter", ["fat"], []), ("cream", ["fat"], []), ("beef", ["fat"], ["protein"]),
         ("cheddar_cheese", ["sodium", "fat"], []), ("rice", [], ["fiber"]),
         ("cane_molasses", ["sugar"], []), ("bacon", ["sodium"], []), ("white_bread", [], ["protein", "fiber"])]

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
BLUE, ORANGE, GREY, GREEN = "#2f6db3", "#e07b39", "#9aa3ad", "#3a9a6b"
CONF_COL = {"High": BLUE, "Medium": "#8fb3de", "Low": "#e3b5a0"}


def upper(S):
    return S[np.triu_indices_from(S, 1)]


def top_sets(model, k=5, **kw):
    return {n: set(model.recommend(n, k, **kw)["substitute"]) for n in model.ingredients["ingredient"]}


# ------------------------------------------------------------------ flavor analyses
def flavor_pair_stats(m):
    cat = m.ingredients["category"].to_numpy()
    iu = np.triu_indices_from(m.S, 1)
    s, same = m.S[iu], cat[iu[0]] == cat[iu[1]]
    return {"pairs": int(len(s)), "zero_pct": float((s == 0).mean() * 100),
            "median": float(np.median(s)), "p95": float(np.percentile(s, 95)),
            "same_category_mean": float(s[same].mean()), "diff_category_mean": float(s[~same].mean())}, s, same


def category_matrix(S, ing):
    cats = sorted(ing["category"].unique())
    M = pd.DataFrame(index=cats, columns=cats, dtype=float)
    for a in cats:
        ia = np.where(ing["category"] == a)[0]
        for b in cats:
            ib = np.where(ing["category"] == b)[0]
            block = S[np.ix_(ia, ib)]
            M.loc[a, b] = np.nanmean(block) if np.isfinite(block).any() else np.nan
    return M


def rec_stats(rec, score):
    top1, top5 = rec[rec["rank"] == 1], rec[rec["rank"] <= 5]
    out = {"ingredients": int(rec["ingredient"].nunique()),
           "top1_same_category_pct": float(top1["same_category"].mean() * 100),
           "top5_same_category_pct": float(top5["same_category"].mean() * 100),
           "top1_score_median": float(top1[score].median()),
           "top5_confidence_pct": (top5["confidence"].value_counts(normalize=True) * 100).round(1).to_dict(),
           "top5_name_nested_pct": float(top5["name_nested"].mean() * 100)}
    if "ci_low" in rec:
        w = (rec["ci_high"] - rec["ci_low"])
        rel = w / rec["similarity"].clip(lower=1e-3)
        out["top5_ci_width_median_by_confidence"] = top5.assign(w=w).groupby("confidence")["w"].median().round(3).to_dict()
        out["top5_relative_ci_width_median_by_confidence"] = top5.assign(r=rel).groupby("confidence")["r"].median().round(2).to_dict()
        out["top5_ci_excludes_zero_pct_by_confidence"] = (top5.assign(e=top5["ci_low"] > 0).groupby("confidence")["e"].mean() * 100).round(1).to_dict()
        out["top5_similarity_median_by_confidence"] = top5.groupby("confidence")["similarity"].median().round(3).to_dict()
        out["top5_stability_median_by_confidence"] = top5.groupby("confidence")["top5_stability"].median().round(3).to_dict()
        out["top1_ci_excludes_zero_pct"] = float((top1["ci_low"] > 0).mean() * 100)
    return out


# ------------------------------------------------------------------ figures
def fig_flavor_overview(m, s, same, M, rec):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    a = ax[0]
    bins = np.linspace(0, 1, 41)
    a.hist(s[~same], bins=bins, color=GREY, density=True, alpha=0.8, label=f"different category (n={(~same).sum():,})")
    a.hist(s[same], bins=bins, color=BLUE, density=True, alpha=0.7, label=f"same category (n={same.sum():,})")
    a.set_yscale("log"); a.set_xlabel("Jaccard similarity"); a.set_ylabel("density (log)")
    a.set_title("A. Pairwise flavor similarity"); a.legend(frameon=False)

    a = ax[1]
    im = a.imshow(M.to_numpy(dtype=float), cmap="Blues")
    a.set_xticks(range(len(M))); a.set_xticklabels(M.columns, rotation=60, ha="right")
    a.set_yticks(range(len(M))); a.set_yticklabels(M.index)
    vmax = np.nanmax(M.to_numpy())
    for i in range(len(M)):
        for j in range(len(M)):
            v = M.iloc[i, j]
            a.text(j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=6, color="white" if v > vmax * 0.6 else "#333")
    a.set_title("B. Mean Jaccard between categories"); fig.colorbar(im, ax=a, fraction=0.04)
    for sp in a.spines.values():
        sp.set_visible(False)

    a = ax[2]
    top1 = rec[rec["rank"] == 1]
    a.scatter(top1["query_compounds"], top1["substitute_compounds"], s=10, color=BLUE, alpha=0.6)
    a.plot([1, 300], [1, 300], color="k", lw=0.8, ls="--")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("compounds in ingredient"); a.set_ylabel("compounds in its top-1 substitute")
    rho = top1["query_compounds"].corr(top1["substitute_compounds"], method="spearman")
    a.set_title(f"C. Profile-size bias (Spearman ρ = {rho:.2f})")
    fig.tight_layout(); fig.savefig(RES / "d2_flavor_overview.png", bbox_inches="tight"); plt.close(fig)
    return float(rho)


def fig_uncertainty(rec):
    top5 = rec[rec["rank"] <= 5].copy()
    top5["rel_width"] = (top5["ci_high"] - top5["ci_low"]) / top5["similarity"].clip(lower=1e-3)
    top5["min_size"] = np.minimum(top5["query_compounds"], top5["substitute_compounds"])
    order = ["High", "Medium", "Low"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    a = ax[0]
    a.scatter(top5["min_size"], top5["rel_width"].clip(upper=6), s=6, alpha=0.35, c=top5["confidence"].map(CONF_COL))
    a.set_xscale("log"); a.set_xlabel("smaller compound profile in the pair")
    a.set_ylabel("relative 95% CI width  (CI width / similarity)")
    a.axvline(5, color="k", lw=0.6, ls=":"); a.axvline(20, color="k", lw=0.6, ls=":")
    a.set_title("A. Relative uncertainty shrinks as profiles grow")
    a.legend(handles=[Patch(color=c, label=l) for l, c in CONF_COL.items()], frameon=False, title="confidence")
    a = ax[1]
    pct = [(top5.loc[top5["confidence"] == c, "ci_low"] > 0).mean() * 100 for c in order]
    a.bar(order, pct, color=[CONF_COL[c] for c in order])
    for x, v in enumerate(pct):
        a.text(x, v + 1.5, f"{v:.0f}%", ha="center")
    a.set_ylim(0, 110); a.set_ylabel("% of top-5 suggestions whose 95% CI excludes 0")
    a.set_title("B. Is the similarity distinguishable from zero?")
    a = ax[2]
    data = [top5.loc[top5["confidence"] == c, "top5_stability"] for c in order]
    bp = a.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for p_, c in zip(bp["boxes"], order):
        p_.set_facecolor(CONF_COL[c])
    a.set_ylabel("share of bootstrap resamples still in top-5"); a.set_ylim(0, 1.02)
    a.set_title("C. Top-5 stability by confidence label")
    fig.tight_layout(); fig.savefig(RES / "d2_flavor_uncertainty.png", bbox_inches="tight"); plt.close(fig)


def fig_cases(f, n):
    fig, axes = plt.subplots(2, len(CASES), figsize=(3.1 * len(CASES), 7.2))
    for j, q in enumerate(CASES):
        a = axes[0, j]
        r = f.recommend(q, 5)[::-1]
        y = np.arange(len(r))
        a.barh(y, r["similarity"], color=[CONF_COL[c] for c in r["confidence"]])
        a.errorbar(r["similarity"], y, xerr=[r["similarity"] - r["ci_low"], r["ci_high"] - r["similarity"]],
                   fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
        a.set_yticks(y); a.set_yticklabels(r["substitute"], fontsize=8); a.set_xlim(0, 1)
        a.set_title(f"{q}\nflavor (Jaccard ± 95% CI)", fontsize=8.5)
        a = axes[1, j]
        if q in n:
            r = n.recommend(q, 5)[::-1]
            a.barh(np.arange(len(r)), r["nutrient_similarity"], color=[CONF_COL[c] for c in r["confidence"]])
            a.set_yticks(np.arange(len(r))); a.set_yticklabels(r["substitute"], fontsize=8)
        a.set_xlim(0, 1); a.set_title("nutrient (1 / (1 + d))", fontsize=8.5)
    fig.legend(handles=[Patch(color=c, label=l) for l, c in CONF_COL.items()], loc="lower center", ncol=3,
               frameon=False, title="confidence")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(RES / "d2_case_studies.png", bbox_inches="tight"); plt.close(fig)


def fig_flavor_vs_nutrient(f, n, nut_rec):
    common = [x for x in n.ingredients["ingredient"] if x in f]
    fi = [f._idx[x] for x in common]
    ni = [n._idx[x] for x in common]
    Sf, Sn = f.S[np.ix_(fi, fi)], n.S[np.ix_(ni, ni)]
    xf, xn = upper(Sf), upper(Sn)
    rho = float(pd.Series(xf).corr(pd.Series(xn), method="spearman"))
    M = category_matrix(n.S, n.ingredients)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    a.hexbin(xf, xn, gridsize=45, bins="log", cmap="Blues", mincnt=1)
    a.set_xlabel("flavor similarity (Jaccard)"); a.set_ylabel("nutrient similarity")
    a.set_title(f"A. Flavor vs nutrient similarity, {len(xf):,} pairs (Spearman ρ = {rho:.2f})")
    a = ax[1]
    im = a.imshow(M.to_numpy(dtype=float), cmap="Greens")
    a.set_xticks(range(len(M))); a.set_xticklabels(M.columns, rotation=60, ha="right")
    a.set_yticks(range(len(M))); a.set_yticklabels(M.index)
    vmax = np.nanmax(M.to_numpy())
    for i in range(len(M)):
        for j in range(len(M)):
            v = M.iloc[i, j]
            if np.isfinite(v):
                a.text(j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=6, color="white" if v > vmax * 0.8 else "#333")
    a.set_title("B. Mean nutrient similarity between categories"); fig.colorbar(im, ax=a, fraction=0.04)
    for sp in a.spines.values():
        sp.set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "d2_flavor_vs_nutrient.png", bbox_inches="tight"); plt.close(fig)
    return rho, M


# ------------------------------------------------------------------ main
def main():
    f = FlavorSubstitutionModel(weighting="jaccard").fit().bootstrap(B=200, k=5)
    fw = FlavorSubstitutionModel(weighting="idf").fit().bootstrap(B=200, k=5)
    n = NutrientSubstitutionModel(features="full").fit()
    nm = NutrientSubstitutionModel(features="macro").fit()
    nall = NutrientSubstitutionModel(features="full", catalog="all").fit()

    rec_f = f.recommend_all(TOP_K)
    rec_fw = fw.recommend_all(TOP_K)
    rec_n = n.recommend_all(TOP_K, flavor_model=f)
    rec_f.to_csv(RES / "recommendations_flavor_jaccard.csv", index=False, float_format="%.4f")
    rec_fw.to_csv(RES / "recommendations_flavor_idf.csv", index=False, float_format="%.4f")
    rec_n.to_csv(RES / "recommendations_nutrient.csv", index=False, float_format="%.4f")
    f.similarity_frame().round(4).to_csv(RES / "similarity_flavor_jaccard.csv")
    n.similarity_frame().round(4).to_csv(RES / "similarity_nutrient.csv")

    rec_nall = nall.recommend_all(TOP_K, flavor_model=f)
    rec_nall.to_csv(RES / "recommendations_nutrient_all_foods.csv", index=False, float_format="%.4f")
    ALL_EX = ["jackfruit", "quinoa", "tofu", "turbot", "chia", "rabbit", "teff", "breadfruit", "butter", "salmon"]
    all_examples = {q: nall.recommend(q, 5)[["substitute", "nutrient_similarity", "category", "origin", "confidence"]]
                    .round(2).values.tolist() for q in ALL_EX if q in nall}
    top1_all = rec_nall[rec_nall["rank"] == 1]
    cat_all = nall.ingredients
    extended = {
        "foods": int(len(cat_all)),
        "recipe_curated": int((cat_all["origin"] == "recipe_curated").sum()),
        "foodb_auto": int((cat_all["origin"] == "foodb_auto").sum()),
        "specialty_not_suggested": int(cat_all["specialty"].sum()),
        "top1_same_food_group_pct": float(top1_all["same_category"].mean() * 100),
        "top5_same_food_group_pct": float(rec_nall[rec_nall["rank"] <= 5]["same_category"].mean() * 100),
        "top5_confidence_pct": (rec_nall[rec_nall["rank"] <= 5]["confidence"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "recipe_queries_top5_from_outside_recipes_pct": float(
            rec_nall[(rec_nall["rank"] <= 5) & rec_nall["ingredient"].isin(n.ingredients["ingredient"])]["origin"]
            .eq("foodb_auto").mean() * 100),
        "examples": all_examples,
    }

    goal_rows = []
    for q, lo, hi in GOALS:
        if q in n:
            r = n.recommend(q, 3, lower=lo, higher=hi, flavor_model=f)
            r.insert(1, "goal", " ".join([f"lower {x}" for x in lo] + [f"higher {x}" for x in hi]))
            goal_rows.append(r)
    goals = pd.concat(goal_rows, ignore_index=True)
    goals.to_csv(RES / "goal_examples.csv", index=False, float_format="%.3f")

    fstats, s, same = flavor_pair_stats(f)
    Mf = category_matrix(f.S, f.ingredients)
    rho_size = fig_flavor_overview(f, s, same, Mf, rec_f)
    fig_uncertainty(rec_f)
    fig_cases(f, n)
    rho_fn, Mn = fig_flavor_vs_nutrient(f, n, rec_n)

    a5, b5 = top_sets(f), top_sets(fw)
    n5, m5 = top_sets(n), top_sets(nm)
    common = [x for x in n5 if x in a5]
    nut_ing = pd.read_csv(PROC / "ingredient_nutrients.csv")

    summary = {
        "flavor": {
            "ingredients_scored": int(len(f.ingredients)),
            "pairs": fstats,
            "category_within_mean": {c: round(float(Mf.loc[c, c]), 3) for c in Mf.index},
            "size_bias_spearman": rho_size,
            "jaccard": rec_stats(rec_f, "similarity"),
            "idf": rec_stats(rec_fw, "similarity"),
            "jaccard_vs_idf_top5_overlap": float(np.mean([len(a5[x] & b5[x]) / 5 for x in a5])),
            "jaccard_vs_idf_spearman": float(pd.Series(upper(f.S)).corr(pd.Series(upper(fw.S)), method="spearman")),
            "confidence_of_ingredients": pd.cut(f.ingredients["n_compounds"], [0, 4, 19, 1e9],
                                                labels=["<5", "5-19", ">=20"]).value_counts().to_dict(),
        },
        "nutrient": {
            "ingredients_scored": int(len(n.ingredients)),
            "features": list(n._cols),
            "full": rec_stats(rec_n, "nutrient_similarity"),
            "full_vs_macro_top5_overlap": float(np.mean([len(n5[x] & m5[x]) / 5 for x in n5])),
            "category_within_mean": {c: round(float(Mn.loc[c, c]), 3) for c in Mn.index},
            "identical_record_in_top5_pct": float(rec_n[rec_n["rank"] <= 5]["identical_record"].mean() * 100),
            "only_nutrient_mode_examples": sorted(set(n.ingredients["ingredient"]) -
                                                  set(f.ingredients.loc[f.ingredients["n_compounds"] >= 5, "ingredient"])),
        },
        "nutrient_extended_catalog": extended,
        "flavor_vs_nutrient": {
            "ingredients_in_both": len(common),
            "pair_similarity_spearman": rho_fn,
            "top5_overlap_mean": float(np.mean([len(a5[x] & n5[x]) / 5 for x in common])),
            "ingredients_with_any_top5_overlap_pct": float(np.mean([len(a5[x] & n5[x]) > 0 for x in common]) * 100),
            "mean_flavor_similarity_of_nutrient_top1": float(rec_n.loc[rec_n["rank"] == 1, "flavor_similarity"].mean()),
            "mean_flavor_similarity_of_flavor_top1": float(rec_f.loc[rec_f["rank"] == 1, "similarity"].mean()),
        },
        "generic_excluded_as_candidates": sorted(GENERIC),
        "case_studies_flavor": {q: f.recommend(q, 5)[["substitute", "similarity", "ci_low", "ci_high", "confidence"]]
                                .round(3).values.tolist() for q in CASES + ["oregano", "chickpea", "vanilla", "onion", "white_wine", "strawberry", "shrimp", "cumin"]},
        "case_studies_nutrient": {q: n.recommend(q, 5, flavor_model=f)[["substitute", "nutrient_similarity", "flavor_similarity", "confidence"]]
                                  .round(3).values.tolist() for q in CASES + ["olive_oil", "honey", "rice", "egg", "milk", "oregano", "lemon_juice"] if q in n},
        "goal_examples": goals[["ingredient", "goal", "rank", "substitute", "nutrient_similarity", "energy_kcal",
                                "protein_g", "fat_g", "sugars_g", "sodium_mg", "flavor_similarity"]].round(2).values.tolist(),
        "nutrient_data_coverage": {"with_nutrients": int((nut_ing["match_type"] != "none").sum())},
    }
    (RES / "d2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ["nutrient_extended_catalog", "flavor_vs_nutrient"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
