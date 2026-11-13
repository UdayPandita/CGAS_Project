"""
Deliverable 4 — Result analysis
===============================

Final analyses that the conclusion rests on (uses the D3 set-up: same candidate pool,
same Recipe1MSubs ground truth, same metrics):

1. Ablation — does adding flavor similarity improve a recommender that already uses
   nutrition and recipe context?
     a) rank-average hybrids with and without flavor, and a confidence-gated hybrid that
        uses flavor only for High-confidence pairs (D2 labels);
     b) a learned ranker (logistic regression, 5-fold cross-validation over *queries*, so
        test ingredients are never seen in training) with and without flavor features.
2. Error analysis — which real substitutions does flavor miss, and why?
     same-category vs cross-category swaps, and the most common category transitions.
3. Overview figure for the final report.

Run:  python3 src/d4_analysis.py      (~30 s)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from d3_evaluation import ROOT, boot_ci, evaluate, paired_diff, pct_rank, setup

RES = ROOT / "results" / "d4"
RES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
BLUE, GREEN, PURPLE, ORANGE, GREY = "#2f6db3", "#3a9a6b", "#8a5cc2", "#e07b39", "#9aa3ad"


def main():
    s = setup()
    names, n, valid, scores, P = s["names"], s["n"], s["valid"], s["scores"], s["P"]
    gt, gt_sets, cat, size, freq, ctx = s["gt"], s["gt_sets"], s["cat"], s["size"], s["freq"], s["ctx"]

    def run(score):
        return evaluate(score, gt_sets, valid, names).set_index("q").sort_index()

    # ------------------------------------------------------------ 1a. rank-average ablation
    Pf, Pn, Pc = P["Flavor (Jaccard)"], P["Nutrient (14 nutrients)"], P["Recipe context"]
    high = np.minimum(size[:, None], size[None, :]) >= 20
    hyb = {
        "Nutrient + context": (Pn + Pc) / 2,
        "Flavor + nutrient + context": (Pf + Pn + Pc) / 3,
        "Confidence-gated: + flavor only if High": np.where(high, (Pf + Pn + Pc) / 3, (Pn + Pc) / 2),
    }
    ab = {k: run(v) for k, v in hyb.items()}
    ablation = {k: {m: boot_ci(e[m]) for m in ["pct", "mrr", "hit@1", "hit@5", "hit@10"]} for k, e in ab.items()}
    ablation_diff = {
        "all three - nutrient+context": {m: paired_diff(ab["Flavor + nutrient + context"][m], ab["Nutrient + context"][m])
                                         for m in ["pct", "mrr", "hit@1", "hit@10"]},
        "gated - nutrient+context": {m: paired_diff(ab["Confidence-gated: + flavor only if High"][m], ab["Nutrient + context"][m])
                                     for m in ["pct", "mrr", "hit@1", "hit@10"]},
        "gated - all three": {m: paired_diff(ab["Confidence-gated: + flavor only if High"][m], ab["Flavor + nutrient + context"][m])
                              for m in ["pct", "mrr", "hit@1", "hit@10"]},
    }

    # ------------------------------------------------------------ 1b. learned ranker, CV over queries
    queries = np.array(sorted(gt_sets))
    rows = [(q, c, int(c in gt_sets[q])) for q in queries for c in np.where(valid[q])[0]]
    pr = pd.DataFrame(rows, columns=["q", "c", "y"])
    F = pd.DataFrame({
        "nutrient": scores["Nutrient (14 nutrients)"][pr.q, pr.c],
        "context": ctx.S[pr.q, pr.c],
        "same_category": (cat[pr.q] == cat[pr.c]).astype(float),
        "popularity": np.log1p(freq[pr.c]),
        "flavor": scores["Flavor (Jaccard)"][pr.q, pr.c],
        "flavor_x_high": scores["Flavor (Jaccard)"][pr.q, pr.c] * high[pr.q, pr.c],
        "high": high[pr.q, pr.c].astype(float),
    })
    feature_sets = {
        "Learned: category + popularity": ["same_category", "popularity"],
        "Learned: + nutrient + context": ["same_category", "popularity", "nutrient", "context"],
        "Learned: + nutrient + context + flavor": ["same_category", "popularity", "nutrient", "context", "flavor"],
        "Learned: + nutrient + context + flavor (confidence-aware)": ["same_category", "popularity", "nutrient", "context",
                                                                       "flavor", "flavor_x_high", "high"],
    }
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(queries), 5)
    learned, coefs = {}, {}
    for name, cols in feature_sets.items():
        S = np.full((n, n), -np.inf)
        cf = []
        for test in folds:
            tr = ~pr["q"].isin(test)
            X = F[cols]
            mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
            lr = LogisticRegression(max_iter=3000).fit((X[tr] - mu) / sd, pr.loc[tr, "y"])
            cf.append(lr.coef_[0])
            te = ~tr
            S[pr.loc[te, "q"], pr.loc[te, "c"]] = lr.decision_function((X[te] - mu) / sd)
        learned[name] = run(S)
        coefs[name] = dict(zip(cols, np.round(np.mean(cf, 0), 3)))
    learned_res = {k: {m: boot_ci(e[m]) for m in ["pct", "mrr", "hit@1", "hit@5", "hit@10"]} for k, e in learned.items()}
    a, b = "Learned: + nutrient + context + flavor", "Learned: + nutrient + context"
    c_ = "Learned: + nutrient + context + flavor (confidence-aware)"
    learned_diff = {
        "+flavor vs without": {m: paired_diff(learned[a][m], learned[b][m]) for m in ["pct", "mrr", "hit@1", "hit@10"]},
        "+flavor (confidence-aware) vs without": {m: paired_diff(learned[c_][m], learned[b][m]) for m in ["pct", "mrr", "hit@1", "hit@10"]},
        "+nutrient+context vs category+popularity": {m: paired_diff(learned[b][m], learned["Learned: category + popularity"][m])
                                                     for m in ["pct", "mrr", "hit@1", "hit@10"]},
    }

    # ------------------------------------------------------------ 2. error analysis
    g = gt.copy()
    g["same_cat"] = cat[g.qi] == cat[g.si]
    for k, col in [("Flavor (Jaccard)", "flavor"), ("Nutrient (14 nutrients)", "nutrient"), ("Recipe context", "context")]:
        g[col] = P[k][g.qi, g.si]
    g["high"] = high[g.qi, g.si]
    g["qcat"], g["scat"] = cat[g.qi], cat[g.si]
    err = {
        "pairs": int(len(g)), "records": int(g["count"].sum()),
        "same_category_share_pairs_pct": float(g["same_cat"].mean() * 100),
        "same_category_share_records_pct": float(g.loc[g.same_cat, "count"].sum() / g["count"].sum() * 100),
        "by_type": {("same category" if sc else "cross category"): {
            "pairs": int(len(d)),
            "flavor_pct": boot_ci(d["flavor"]), "nutrient_pct": boot_ci(d["nutrient"]), "context_pct": boot_ci(d["context"]),
            "flavor_pct_high_conf_only": boot_ci(d.loc[d.high, "flavor"]) if d.high.any() else None}
            for sc, d in g.groupby("same_cat")},
    }
    cross = g[~g.same_cat].groupby(["qcat", "scat"]).agg(records=("count", "sum"), pairs=("count", "size"),
                                                        flavor_pct=("flavor", "mean"), nutrient_pct=("nutrient", "mean"))
    cross = cross.sort_values("records", ascending=False).round(3)
    cross.to_csv(RES / "cross_category_transitions.csv")
    top_cross = g[~g.same_cat].sort_values("count", ascending=False).head(15)[["query", "sub", "count", "flavor", "nutrient"]].round(2)
    err["top_cross_category_swaps"] = top_cross.values.tolist()
    err["top_transitions"] = cross.head(10).reset_index().values.tolist()

    # how often does flavor rank a real substitute in the bottom half? and what are those cases?
    worst = g.sort_values(["flavor", "count"], ascending=[True, False])
    worst = worst[worst["count"] >= 10].head(15)[["query", "sub", "count", "flavor", "nutrient", "same_cat", "high"]].round(2)
    err["frequent_swaps_flavor_ranks_low"] = worst.values.tolist()
    best = g[g["count"] >= 10].sort_values("flavor", ascending=False).head(15)[["query", "sub", "count", "flavor", "nutrient"]].round(2)
    err["frequent_swaps_flavor_ranks_high"] = best.values.tolist()

    # ------------------------------------------------------------ 3. overview figure
    d3 = json.loads((ROOT / "results" / "d3" / "d3_summary.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    a_ = ax[0, 0]
    ladder = [("Random", "Baseline: random", GREY), ("Profile-size match", "Baseline: profile-size match", GREY),
              ("Flavor (Jaccard)", "Flavor (Jaccard)", BLUE), ("Same category", "Baseline: same category", GREY),
              ("Nutrient", "Nutrient (14 nutrients)", GREEN), ("Recipe context", "Recipe context", PURPLE),
              ("Flavor + nutrient + context", "Hybrid: all three", ORANGE),
              ("Same category + popularity", "Baseline: same category + popularity", GREY)]
    v = np.array([d3["metrics"][k]["pct"] for _, k, _ in ladder])
    a_.barh([l for l, _, _ in ladder], v[:, 0], color=[c for *_, c in ladder],
            xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], capsize=2)
    for y, (x, hi) in enumerate(zip(v[:, 0], v[:, 2])):
        a_.text(hi + 0.01, y, f"{x:.2f}", va="center", fontsize=8)
    a_.axvline(0.5, color="k", ls=":", lw=0.8); a_.set_xlim(0.4, 1)
    a_.set_xlabel("mean percentile rank of real substitutes (0.5 = random)")
    a_.set_title("A. Predicting real substitutions (Recipe1MSubs)")

    a_ = ax[0, 1]
    conf = d3["confidence_gt_percentile"]
    order = ["High", "Medium", "Low"]
    vv = np.array([conf[o][:3] for o in order])
    a_.bar(order, vv[:, 0], color=[BLUE, "#8fb3de", "#e3b5a0"], yerr=[vv[:, 0] - vv[:, 1], vv[:, 2] - vv[:, 0]], capsize=3)
    for x, (val, hi) in enumerate(zip(vv[:, 0], vv[:, 2])):
        a_.text(x, hi + 0.01, f"{val:.2f}", ha="center")
    a_.axhline(0.5, color="k", ls=":", lw=0.8); a_.set_ylim(0.3, 0.9)
    a_.set_ylabel("flavor: mean percentile of real substitutes")
    a_.set_title("B. Flavor similarity works when both compound profiles are rich")

    a_ = ax[1, 0]
    t = err["by_type"]
    groups = ["same category", "cross category"]
    w = 0.25
    for j, (m, col, lab) in enumerate([("flavor_pct", BLUE, "flavor"), ("nutrient_pct", GREEN, "nutrient"),
                                       ("context_pct", PURPLE, "recipe context")]):
        vals = np.array([t[gname][m] for gname in groups])
        a_.bar(np.arange(2) + (j - 1) * w, vals[:, 0], w, color=col, label=lab,
               yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], capsize=2)
    a_.set_xticks(range(2))
    a_.set_xticklabels([f"{gname}\n({t[gname]['pairs']} pairs)" for gname in groups])
    a_.axhline(0.5, color="k", ls=":", lw=0.8); a_.set_ylim(0.3, 1)
    a_.set_ylabel("mean percentile of real substitutes"); a_.legend(frameon=False)
    a_.set_title("C. Cross-category (functional) swaps defeat flavor similarity")

    a_ = ax[1, 1]
    labs = ["category\n+ popularity", "+ nutrient\n+ context", "+ flavor", "+ flavor\n(confidence-\naware)"]
    keys = list(feature_sets)
    vv = np.array([learned_res[k]["hit@10"] for k in keys]) * 100
    a_.bar(labs, vv[:, 0], color=[GREY, GREEN, BLUE, BLUE], yerr=[vv[:, 0] - vv[:, 1], vv[:, 2] - vv[:, 0]], capsize=3)
    for x, (val, hi) in enumerate(zip(vv[:, 0], vv[:, 2])):
        a_.text(x, hi + 1, f"{val:.1f}%", ha="center")
    a_.set_ylabel("Hit@10 (%) on held-out ingredients"); a_.set_ylim(50, 100)
    a_.set_title("D. Learned ranker (5-fold CV): what does each signal add?")
    fig.tight_layout()
    fig.savefig(RES / "d4_overview.png", bbox_inches="tight")
    plt.close(fig)

    out = {"ablation_rank_average": ablation, "ablation_rank_average_diff": ablation_diff,
           "learned_ranker": learned_res, "learned_ranker_diff": learned_diff, "learned_coefficients": coefs,
           "error_analysis": err}
    (RES / "d4_summary.json").write_text(json.dumps(out, indent=1, default=str))
    r3 = lambda t: [round(x, 3) for x in t]
    for k, v in ablation.items():
        print("RANKAVG", k, {m: r3(x) for m, x in v.items()})
    for k, v in ablation_diff.items():
        print("RANKAVG DIFF", k, {m: r3(x) for m, x in v.items()})
    for k, v in learned_res.items():
        print("LEARNED", k, {m: r3(x) for m, x in v.items()})
    for k, v in learned_diff.items():
        print("LEARNED DIFF", k, {m: r3(x) for m, x in v.items()})
    print("COEFS", coefs)
    print(json.dumps({k: v for k, v in err.items() if k not in ("by_type",)}, indent=0, default=str)[:3500])
    print({k: {m: (r3(x) if isinstance(x, tuple) else x) for m, x in v.items()} for k, v in err["by_type"].items()})


if __name__ == "__main__":
    main()
