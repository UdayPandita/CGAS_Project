"""
Deliverable 3 — Recommendation Evaluation
=========================================

Two independent sources of evidence about whether a suggested substitute is plausible:

A. Real substitutions (ground truth): Recipe1MSubs (Fatemi et al. 2023), substitutions that
   cooks reported in recipe comments, e.g. "used olive oil instead of butter". Recipe1M names
   are mapped onto our ingredient names (src/recipe1m_mapping.py). Metrics: Hit@k, P@k, R@k,
   MRR and the mean percentile rank of the true substitutes (0.5 = random, 1 = always first).

B. Recipe co-occurrence (the 56k flavor-network recipes): real substitutes should
     * appear in the same kinds of dishes  -> high *shared-context* similarity
       (cosine of PPMI co-occurrence vectors with the pair's own entries removed), and
     * rarely appear together               -> *exclusivity*, lift = observed / expected
       co-occurrence <= 1 (complements such as butter + flour have lift > 1).

All methods rank the same candidate pool: recipe ingredients that have both flavor and
nutrient data, minus generic catch-all terms. Forms of the query (beef -> roasted_beef) are
never counted. Ties get average ranks. 95% CIs come from bootstrapping over queries.

Run:  python3 src/d3_evaluation.py         (~1 min)
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
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

import safe_pickle
from recipe1m_mapping import NameMapper
from substitution_model import (GENERIC, PROC, ROOT, FlavorSubstitutionModel,
                                NutrientSubstitutionModel, is_variant)

RES = ROOT / "results" / "d3"
RES.mkdir(parents=True, exist_ok=True)
SUBS_DIR = ROOT / "data" / "raw" / "recipe1msubs"
RNG = np.random.default_rng(0)
N_BOOT = 1000
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
COL = {"flavor": "#2f6db3", "nutrient": "#3a9a6b", "context": "#8a5cc2", "hybrid": "#e07b39", "baseline": "#9aa3ad"}


# ====================================================================== data
def load_ground_truth(names):
    files = sorted(SUBS_DIR.glob("*_comments_subs.pkl"))
    if not files:
        raise SystemExit(f"No Recipe1MSubs files in {SUBS_DIR}")
    mapper = NameMapper(names)
    raw, kept = 0, Counter()
    all_names = Counter()
    for f in files:
        for rec in safe_pickle.load(f):
            a, b = rec["subs"]
            raw += 1
            all_names[a] += 1
            all_names[b] += 1
            ma, mb = mapper.map(a), mapper.map(b)
            if ma and mb and ma != mb and not is_variant(ma, mb):
                kept[(ma, mb)] += 1
    mapped_mentions = sum(k for n, k in all_names.items() if mapper.map(n))
    info = {"files": [f.name for f in files], "records": raw, "distinct_names": len(all_names),
            "names_mapped": sum(1 for n in all_names if mapper.map(n)),
            "name_mentions_mapped_pct": mapped_mentions / sum(all_names.values()) * 100,
            "records_mapped_pairs": sum(kept.values()), "unique_directed_pairs": len(kept)}
    gt = pd.DataFrame([(a, b, k) for (a, b), k in kept.items()], columns=["query", "sub", "count"])
    return gt, info


class RecipeContext:
    """Shared-context similarity and lift from the recipe x ingredient matrix."""

    def __init__(self, names, recipe_mask=None):
        R = sparse.load_npz(PROC / "recipe_ingredient_matrix.npz").astype(np.float64).tocsr()
        ing = pd.read_csv(PROC / "ingredients.csv").set_index("ingredient")
        if recipe_mask is not None:
            R = R[recipe_mask]
        Ru = R[:, ing.loc[names, "row"].to_numpy()]
        n = Ru.shape[0]
        C = (Ru.T @ Ru).toarray()
        f = np.diag(C).copy()
        np.fill_diagonal(C, 0)
        E = np.outer(f, f) / n
        self.lift = np.divide(C, E, out=np.zeros_like(C), where=E > 0)
        self.cooc = C
        self.freq = f
        pmi = np.log(np.divide(C, E, out=np.ones_like(C), where=(E > 0) & (C > 0)))
        ppmi = np.maximum(pmi, 0)
        # cosine with the pair's own entries removed: ppmi[a,a] = ppmi[b,b] = 0, so the dot
        # product never includes them; remove ppmi[a,b] from |a| and ppmi[b,a] from |b|.
        sq = (ppmi ** 2).sum(1)
        na = np.sqrt(np.maximum(sq[:, None] - ppmi ** 2, 0))
        nb = np.sqrt(np.maximum(sq[None, :] - ppmi.T ** 2, 0))
        self.S = np.divide(ppmi @ ppmi.T, na * nb, out=np.zeros_like(C), where=(na * nb) > 0)
        np.fill_diagonal(self.S, np.nan)


# ====================================================================== ranking metrics
def evaluate(score, gt_sets, valid, names, ks=(1, 5, 10)):
    """score: n x n matrix (higher = better); valid: n x n bool mask of allowed candidates.
    Returns per-query metrics DataFrame."""
    rows = []
    for qi, subs in gt_sets.items():
        cand = np.where(valid[qi])[0]
        s = score[qi, cand]
        s = np.where(np.isfinite(s), s, -np.inf)
        rank = rankdata(-s, method="average")                    # 1 = best
        order = cand[np.lexsort((names[cand], -s))]              # ties alphabetical (as in the CLI)
        pos = {c: r for c, r in zip(cand, rank)}
        gt_r = np.array([pos[t] for t in subs if t in pos])
        if len(gt_r) == 0:
            continue
        n = len(cand)
        row = {"q": qi, "n_gt": len(gt_r), "n_cand": n,
               "mrr": 1 / gt_r.min(),
               "pct": float(np.mean(1 - (gt_r - 1) / (n - 1)))}
        for k in ks:
            top = set(order[:k])
            hits = len(top & set(subs))
            row[f"hit@{k}"] = float(hits > 0)
            row[f"p@{k}"] = hits / k
            row[f"r@{k}"] = hits / len(gt_r)
        rows.append(row)
    return pd.DataFrame(rows)


def boot_ci(x, n=N_BOOT):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_diff(a, b, n=N_BOOT):
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pct_rank(score, valid):
    """Within-query percentile rank (1 = best candidate) of every candidate."""
    P = np.full(score.shape, np.nan)
    for i in range(score.shape[0]):
        c = np.where(valid[i])[0]
        s = np.where(np.isfinite(score[i, c]), score[i, c], -np.inf)
        P[i, c] = 1 - (rankdata(-s, method="average") - 1) / (len(c) - 1)
    return P


# ====================================================================== main
def setup():
    """Models, common candidate pool, score matrices and mapped ground truth (shared with D4)."""
    f = FlavorSubstitutionModel().fit()
    fw = FlavorSubstitutionModel(weighting="idf").fit()
    nfull = NutrientSubstitutionModel(features="full").fit()
    nmac = NutrientSubstitutionModel(features="macro").fit()

    pool = [x for x in nfull.ingredients["ingredient"] if x in f and x not in GENERIC]
    names = np.array(pool)
    n = len(names)
    ix = {x: i for i, x in enumerate(names)}

    def sub(model, S):
        j = [model._idx[x] for x in names]
        return S[np.ix_(j, j)]

    ctx = RecipeContext(list(names))
    ing = pd.read_csv(PROC / "ingredients.csv").set_index("ingredient").loc[names]
    size = ing["n_compounds"].to_numpy()
    cat = ing["category"].to_numpy()
    freq = ing["recipe_freq"].to_numpy().astype(float)

    variant = np.array([[is_variant(a, b) for b in names] for a in names])
    valid = ~variant
    np.fill_diagonal(valid, False)

    scores = {
        "Flavor (Jaccard)": sub(f, f.S),
        "Flavor (IDF-Jaccard)": sub(fw, fw.S),
        "Nutrient (14 nutrients)": sub(nfull, nfull.S),
        "Nutrient (macros)": sub(nmac, nmac.S),
        "Recipe context": ctx.S,
    }
    P = {k: pct_rank(v, valid) for k, v in scores.items()}
    scores["Hybrid: flavor + nutrient"] = (P["Flavor (Jaccard)"] + P["Nutrient (14 nutrients)"]) / 2
    scores["Hybrid: flavor + context"] = (P["Flavor (Jaccard)"] + P["Recipe context"]) / 2
    scores["Hybrid: all three"] = (P["Flavor (Jaccard)"] + P["Nutrient (14 nutrients)"] + P["Recipe context"]) / 3
    # baselines
    scores["Baseline: random"] = RNG.random((n, n))
    scores["Baseline: same category"] = (cat[:, None] == cat[None, :]).astype(float) + 1e-6 * RNG.random((n, n))
    scores["Baseline: popularity"] = np.tile(np.log1p(freq), (n, 1))
    scores["Baseline: profile-size match"] = -np.abs(np.log(size)[:, None] - np.log(size)[None, :])
    scores["Baseline: same category + popularity"] = (cat[:, None] == cat[None, :]) * 100 + np.tile(np.log1p(freq), (n, 1))

    # ---------------------------------------------------------------- ground truth
    gt, gt_info = load_ground_truth(f.ingredients["ingredient"])
    gt = gt[gt["query"].isin(ix) & gt["sub"].isin(ix)].copy()
    gt["qi"] = gt["query"].map(ix)
    gt["si"] = gt["sub"].map(ix)
    gt_info.update({"pairs_in_pool": int(len(gt)), "records_in_pool": int(gt["count"].sum()),
                    "queries_in_pool": int(gt["query"].nunique()), "candidate_pool": n})
    gt_sets = gt.groupby("qi")["si"].apply(set).to_dict()
    gt_sets2 = gt[gt["count"] >= 2].groupby("qi")["si"].apply(set).to_dict()
    gt.to_csv(RES / "ground_truth_pairs.csv", index=False)
    return dict(f=f, fw=fw, nfull=nfull, nmac=nmac, names=names, n=n, ix=ix, ctx=ctx, size=size, cat=cat,
                freq=freq, valid=valid, scores=scores, P=P, gt=gt, gt_info=gt_info, gt_sets=gt_sets,
                gt_sets2=gt_sets2)


def main():
    s = setup()
    (f, fw, nfull, nmac, names, n, ix, ctx, size, cat, freq, valid, scores, P, gt, gt_info, gt_sets,
     gt_sets2) = (s[k] for k in ["f", "fw", "nfull", "nmac", "names", "n", "ix", "ctx", "size", "cat", "freq",
                                 "valid", "scores", "P", "gt", "gt_info", "gt_sets", "gt_sets2"])

    results, perq = {}, {}
    for k, s in scores.items():
        e = evaluate(s, gt_sets, valid, names)
        perq[k] = e
        e2 = evaluate(s, gt_sets2, valid, names)
        results[k] = {m: boot_ci(e[m]) for m in ["pct", "mrr", "hit@1", "hit@5", "hit@10", "p@5", "r@10"]}
        results[k]["pct_count>=2"] = boot_ci(e2["pct"])
        results[k]["hit@10_count>=2"] = boot_ci(e2["hit@10"])
    table = pd.DataFrame({k: {m: v[0] for m, v in r.items()} for k, r in results.items()}).T
    table.to_csv(RES / "gt_metrics.csv", float_format="%.4f")

    base = "Baseline: random"
    comparisons = {}
    for a, b in [("Flavor (Jaccard)", "Baseline: random"), ("Flavor (Jaccard)", "Baseline: same category"),
                 ("Flavor (Jaccard)", "Baseline: profile-size match"), ("Flavor (Jaccard)", "Nutrient (14 nutrients)"),
                 ("Flavor (Jaccard)", "Recipe context"), ("Flavor (Jaccard)", "Flavor (IDF-Jaccard)"),
                 ("Hybrid: all three", "Flavor (Jaccard)"), ("Hybrid: flavor + nutrient", "Flavor (Jaccard)"),
                 ("Nutrient (14 nutrients)", "Baseline: same category"),
                 ("Flavor (Jaccard)", "Baseline: same category + popularity")]:
        comparisons[f"{a} - {b}"] = {m: paired_diff(perq[a].set_index("q").loc[perq[b]["q"], m], perq[b][m])
                                     for m in ["pct", "hit@10"]}

    # ---------------------------------------------------------------- does chemistry add information?
    qs = list(gt_sets)
    rows = []
    for qi in qs:
        for ci in np.where(valid[qi])[0]:
            rows.append((qi, ci, int(ci in gt_sets[qi])))
    pr = pd.DataFrame(rows, columns=["qi", "ci", "y"])
    feat = pd.DataFrame({
        "flavor_jaccard": scores["Flavor (Jaccard)"][pr.qi, pr.ci],
        "nutrient_sim": scores["Nutrient (14 nutrients)"][pr.qi, pr.ci],
        "context_sim": ctx.S[pr.qi, pr.ci],
        "same_category": (cat[pr.qi] == cat[pr.ci]).astype(float),
        "size_match": scores["Baseline: profile-size match"][pr.qi, pr.ci],
        "popularity": np.log1p(freq[pr.ci]),
    })
    Z = (feat - feat.mean()) / feat.std()
    coefs = []
    qarr = pr["qi"].to_numpy()
    uq = np.unique(qarr)
    groups = {q: np.where(qarr == q)[0] for q in uq}
    for b in range(200):
        sel = np.concatenate([groups[q] for q in RNG.choice(uq, len(uq))])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Z.iloc[sel], pr["y"].iloc[sel])
        coefs.append(lr.coef_[0])
    coefs = np.array(coefs)
    lr_full = LogisticRegression(max_iter=2000).fit(Z, pr["y"])
    logit = {c: {"coef": float(lr_full.coef_[0][j]), "ci": [float(np.percentile(coefs[:, j], 2.5)),
                                                            float(np.percentile(coefs[:, j], 97.5))]}
             for j, c in enumerate(Z.columns)}

    # within-category test: among SAME-category candidates only, does Jaccard still rank true subs higher?
    same = (cat[:, None] == cat[None, :]) & valid
    within = {}
    for k in ["Flavor (Jaccard)", "Nutrient (14 nutrients)", "Recipe context", "Baseline: random"]:
        e = evaluate(scores[k], {q: s for q, s in gt_sets.items()}, same, names)
        within[k] = boot_ci(e["pct"]) if len(e) else None
        within[k + " (queries)"] = int(len(e))

    # ---------------------------------------------------------------- confidence labels (flavor)
    lab = np.where(np.minimum(size[:, None], size[None, :]) >= 20, "High",
                   np.where(np.minimum(size[:, None], size[None, :]) >= 5, "Medium", "Low"))
    Pf = P["Flavor (Jaccard)"]
    gtp = gt.assign(label=lab[gt.qi, gt.si], pct=Pf[gt.qi, gt.si])
    conf_gt_pct = {l: boot_ci(g["pct"]) + (int(len(g)),) for l, g in gtp.groupby("label")}
    # precision of top-5 flavor suggestions, grouped by the suggestion's label
    sug = []
    S_f = scores["Flavor (Jaccard)"]
    for qi in gt_sets:
        c = np.where(valid[qi])[0]
        top = c[np.lexsort((names[c], -np.nan_to_num(S_f[qi, c], nan=-1)))][:5]
        for t in top:
            sug.append((qi, t, lab[qi, t], int(t in gt_sets[qi])))
    sug = pd.DataFrame(sug, columns=["qi", "ci", "label", "hit"])
    conf_prec = {l: boot_ci(g["hit"]) + (int(len(g)),) for l, g in sug.groupby("label")}

    # ---------------------------------------------------------------- category breakdown
    qcat = pd.Series(cat[list(gt_sets)], index=list(gt_sets))
    by_cat = {}
    for k in ["Flavor (Jaccard)", "Nutrient (14 nutrients)", "Recipe context", "Hybrid: all three"]:
        e = perq[k].assign(cat=lambda d: qcat.loc[d["q"]].to_numpy())
        by_cat[k] = e.groupby("cat").agg(n=("pct", "size"), pct=("pct", "mean"), hit10=("hit@10", "mean")).round(3)
    cat_tab = pd.concat(by_cat, axis=1)
    cat_tab.to_csv(RES / "gt_by_category.csv")

    # ---------------------------------------------------------------- B. co-occurrence validation
    Pc = pct_rank(ctx.S, valid)
    reliable = ctx.freq >= 20                               # enough recipes for a context vector
    def cooc_profile(pairs):
        q, c = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        ok = reliable[q] & reliable[c]
        q, c = q[ok], c[ok]
        return {"pairs": int(len(q)),
                "context_pct": boot_ci(Pc[q, c]),
                "exclusive_pct": boot_ci((ctx.lift[q, c] <= 1).astype(float) * 100),
                "never_together_pct": boot_ci((ctx.cooc[q, c] == 0).astype(float) * 100),
                "median_lift": float(np.median(ctx.lift[q, c]))}

    def top5_pairs(S):
        out = []
        for qi in range(n):
            c = np.where(valid[qi])[0]
            top = c[np.lexsort((names[c], -np.nan_to_num(S[qi, c], nan=-1)))][:5]
            out += [(qi, t) for t in top]
        return out

    cooc = {"Real substitutes (Recipe1MSubs)": cooc_profile(list(zip(gt.qi, gt.si)))}
    for k in ["Flavor (Jaccard)", "Flavor (IDF-Jaccard)", "Nutrient (14 nutrients)", "Hybrid: flavor + nutrient",
              "Baseline: same category", "Baseline: random"]:
        cooc[k + " top-5"] = cooc_profile(top5_pairs(scores[k]))
    allp = [(i, j) for i in range(n) for j in range(n) if valid[i, j]]
    cooc["All pairs"] = cooc_profile(allp)

    # chemistry vs recipe context over all pairs
    iu = np.triu_indices(n, 1)
    ok = reliable[iu[0]] & reliable[iu[1]]
    rho_ctx = spearmanr(scores["Flavor (Jaccard)"][iu][ok], ctx.S[iu][ok]).statistic
    rho_ctx_same = spearmanr(scores["Flavor (Jaccard)"][iu][ok & (cat[iu[0]] == cat[iu[1]])],
                             ctx.S[iu][ok & (cat[iu[0]] == cat[iu[1]])]).statistic
    rho_lift = spearmanr(scores["Flavor (Jaccard)"][iu][ok], np.log1p(ctx.lift[iu][ok])).statistic

    # ---------------------------------------------------------------- cuisine robustness
    rec = pd.read_csv(PROC / "recipes.csv")
    na_mask = (rec["cuisine"] == "NorthAmerican").to_numpy()
    cuisine = {}
    for label, m in [("North American recipes", na_mask), ("All other cuisines", ~na_mask)]:
        cc = RecipeContext(list(names), m)
        Pcc = pct_rank(cc.S, valid)
        rel = cc.freq >= 20
        tp = top5_pairs(scores["Flavor (Jaccard)"])
        q = np.array([p[0] for p in tp]); c = np.array([p[1] for p in tp]); k = rel[q] & rel[c]
        g = rel[gt.qi.to_numpy()] & rel[gt.si.to_numpy()]
        cuisine[label] = {"recipes": int(m.sum()),
                          "flavor_top5_context_pct": float(np.nanmean(Pcc[q[k], c[k]])),
                          "real_subs_context_pct": float(np.nanmean(Pcc[gt.qi.to_numpy()[g], gt.si.to_numpy()[g]]))}

    # ---------------------------------------------------------------- examples
    ex = []
    for qn in ["butter", "onion", "walnut", "lemon_juice", "beef", "milk", "parsley", "cream", "white_wine", "honey",
               "chicken", "vegetable_oil", "raisin", "cilantro"]:
        if qn not in ix:
            continue
        qi = ix[qn]
        truth = gt[gt.qi == qi].sort_values("count", ascending=False)
        row = {"ingredient": qn, "real_substitutes (count)": ", ".join(f"{s}({c})" for s, c in zip(truth["sub"].head(5), truth["count"].head(5)))}
        for k in ["Flavor (Jaccard)", "Nutrient (14 nutrients)", "Hybrid: all three"]:
            c = np.where(valid[qi])[0]
            top = c[np.lexsort((names[c], -np.nan_to_num(scores[k][qi, c], nan=-1)))][:5]
            row[k] = ", ".join(("*" if t in gt_sets.get(qi, set()) else "") + names[t] for t in top)
        ex.append(row)
    pd.DataFrame(ex).to_csv(RES / "examples.csv", index=False)

    # ---------------------------------------------------------------- figures
    fig_methods(results)
    fig_confidence(conf_gt_pct, conf_prec)
    fig_cooc(cooc)
    fig_category(cat_tab)

    summary = {"ground_truth": gt_info, "metrics": results, "paired_differences": comparisons,
               "logistic_regression_standardised": logit, "logit_pairs": int(len(pr)), "logit_positives": int(pr.y.sum()),
               "within_same_category_pct": within, "confidence_gt_percentile": conf_gt_pct,
               "confidence_top5_precision": conf_prec, "cooccurrence": cooc,
               "spearman_jaccard_vs_context": float(rho_ctx), "spearman_jaccard_vs_context_same_category": float(rho_ctx_same),
               "spearman_jaccard_vs_lift": float(rho_lift), "cuisine_robustness": cuisine,
               "examples": ex}
    (RES / "d3_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(json.dumps(gt_info, indent=1))
        print(table[["pct", "mrr", "hit@1", "hit@5", "hit@10", "p@5", "r@10", "pct_count>=2"]].round(3))


# ====================================================================== figures
def _color(name):
    for k, c in [("Flavor", COL["flavor"]), ("Nutrient", COL["nutrient"]), ("Recipe context", COL["context"]),
                 ("Hybrid", COL["hybrid"])]:
        if name.startswith(k):
            return c
    return COL["baseline"]


def fig_methods(results):
    order = sorted(results, key=lambda k: results[k]["pct"][0])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for a, m, lab in [(ax[0], "pct", "mean percentile rank of real substitutes\n(0.5 = random, 1 = always ranked first)"),
                      (ax[1], "hit@10", "Hit@10: share of ingredients with a real substitute in the top 10")]:
        v = np.array([results[k][m] for k in order])
        a.barh(order, v[:, 0], color=[_color(k) for k in order])
        a.errorbar(v[:, 0], range(len(order)), xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="#333", capsize=2, lw=0.8)
        for y, (x, hi) in enumerate(zip(v[:, 0], v[:, 2])):
            a.text(hi + 0.01, y, f"{x:.2f}", va="center", fontsize=8)
        a.set_xlabel(lab)
        if m == "pct":
            a.axvline(0.5, color="k", ls=":", lw=0.8); a.set_xlim(0.4, 1.0)
        else:
            a.set_xlim(0, 1)
    ax[0].set_title("A. How highly are real substitutes ranked?"); ax[1].set_title("B. Hit@10")
    fig.tight_layout(); fig.savefig(RES / "d3_methods.png", bbox_inches="tight"); plt.close(fig)


def fig_confidence(gtpct, prec):
    order = ["High", "Medium", "Low"]
    cmap = {"High": "#2f6db3", "Medium": "#8fb3de", "Low": "#e3b5a0"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, d, lab, scale in [(ax[0], gtpct, "mean percentile rank of real substitutes", 1),
                             (ax[1], prec, "precision of top-5 flavor suggestions (%)", 100)]:
        v = np.array([d[o][:3] for o in order]) * scale
        a.bar(order, v[:, 0], color=[cmap[o] for o in order], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=3)
        for x, (o, val) in enumerate(zip(order, v[:, 0])):
            a.text(x, val + (0.01 if scale == 1 else 0.6), f"{val:.2f}" if scale == 1 else f"{val:.1f}%\n(n={d[o][3]})",
                   ha="center", fontsize=8)
        a.set_ylabel(lab)
    ax[0].axhline(0.5, color="k", ls=":", lw=0.8); ax[0].set_ylim(0.3, 1)
    ax[0].set_title("A. Real substitute pairs, by flavor confidence")
    ax[1].set_title("B. Flavor suggestions confirmed by real cooks")
    fig.tight_layout(); fig.savefig(RES / "d3_confidence.png", bbox_inches="tight"); plt.close(fig)


def fig_cooc(cooc):
    keys = list(cooc)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for a, m, lab in [(ax[0], "context_pct", "shared-context percentile (same kinds of dishes)"),
                      (ax[1], "exclusive_pct", "% of pairs that co-occur no more than chance (lift <= 1)")]:
        v = np.array([cooc[k][m] for k in keys])
        colors = ["#222" if k.startswith("Real") else _color(k) for k in keys]
        a.barh(keys, v[:, 0], color=colors, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=2)
        for y, (x, hi) in enumerate(zip(v[:, 0], v[:, 2])):
            a.text(hi + (0.01 if m == "context_pct" else 1), y, f"{x:.2f}" if m == "context_pct" else f"{x:.0f}%", va="center", fontsize=8)
        a.set_xlabel(lab)
    ax[0].axvline(0.5, color="k", ls=":", lw=0.8); ax[0].set_xlim(0.3, 1)
    ax[0].set_title("A. Do suggested pairs appear in the same kinds of dishes?")
    ax[1].set_title("B. Are suggested pairs used instead of each other?")
    ax[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(RES / "d3_cooccurrence.png", bbox_inches="tight"); plt.close(fig)


def fig_category(cat_tab):
    t = cat_tab.xs("pct", axis=1, level=1)
    n = cat_tab.xs("n", axis=1, level=1).iloc[:, 0]
    t = t[n >= 5]
    t.index = [f"{c} (n={n[c]})" for c in t.index]
    fig, a = plt.subplots(figsize=(8, 5))
    im = a.imshow(t.to_numpy(), cmap="RdBu", vmin=0.3, vmax=0.9, aspect="auto")
    a.set_xticks(range(t.shape[1])); a.set_xticklabels(t.columns, rotation=20, ha="right")
    a.set_yticks(range(len(t))); a.set_yticklabels(t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            a.text(j, i, f"{t.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=a, label="mean percentile rank of real substitutes")
    a.set_title("Performance by ingredient category (categories with >= 5 query ingredients)")
    for sp in a.spines.values():
        sp.set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "d3_by_category.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
