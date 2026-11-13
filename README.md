# Chemistry-Based Ingredient Substitution (CGAS project)

Can chemical flavor similarity predict plausible ingredient substitutions? Ingredients are
represented by their flavor compounds (and, as a second mode, by their nutrients). Substitutes
are ranked by similarity, and the suggestions are validated against recipe co-occurrence and
against real substitutions reported by cooks.

**Answer in one line:** partly. Flavor similarity works well for same-family swaps between
well-characterised ingredients, but fails for sparse compound data and for functional swaps
(fat, acid, liquid). See `reports/D4_results_and_conclusion.md`.

## Reports (one per deliverable)
| | |
|---|---|
| D1 Data pipeline & ingredient–compound matrix (+ FooDB nutrients) | `reports/D1_data_pipeline.md` |
| D2 Flavor- and nutrient-based substitution models | `reports/D2_substitution_model.md` |
| D3 Recommendation evaluation | `reports/D3_evaluation.md` |
| D4 Result analysis & conclusion | `reports/D4_results_and_conclusion.md` |

## Try the recommender
```
python3 src/recommend.py butter                                        # flavor mode (with 95% CI + confidence)
python3 src/recommend.py butter --mode nutrient --lower fat             # nutrient mode with a goal
python3 src/recommend.py beef   --mode nutrient --lower fat --higher protein
python3 src/recommend.py quinoa --mode nutrient                         # any of 808 FooDB foods
python3 src/recommend.py --list [--mode nutrient]                       # all queryable names
python3 src/recommend.py -h                                             # all options
```

## Website and API (Flavor Swap)
A small web app for finding an alternative ingredient, by shared flavor chemistry or by nutrition
(with nutrient goals such as lower fat, confidence intervals, shared compounds, and the substitutions
real cooks reported).

```
pip install -r webapp/requirements.txt
python3 webapp/app.py                     # opens the site in Chrome
```
Deploy: the `Dockerfile` builds a self-contained image from the processed outputs only
(`docker build -t flavor-swap . && docker run -p 8000:8000 flavor-swap`). On Render, push the
repository to GitHub and create a Blueprint from it; `render.yaml` is already configured.
Run the pipeline below first so that `data/processed/` and `results/` exist.

## Reproduce everything
```
pip install -r requirements.txt
python3 src/d1_data_pipeline.py        # D1a  flavor matrix                     (~5 s)
python3 src/foodb_extract.py           # D1b  stream FooDB Content.json once     (~20 s)
python3 src/d1_nutrient_pipeline.py    # D1b  nutrient matrix + 808-food catalog (~5 s)
python3 src/d2_substitution_model.py   # D2   models, recommendations, figures   (~15 s)
python3 src/d3_evaluation.py           # D3   evaluation                         (~15 s)
python3 src/d4_analysis.py             # D4   ablation + error analysis          (~5 s)
```
All scripts are deterministic (fixed seeds).

## Data (in `data/raw/`, not redistributed)
| Source | Files | Use |
|---|---|---|
| Ahn et al. 2011, *Flavor network and the principles of food pairing*, Sci. Rep. 1:196 | `ingr_info.tsv`, `comp_info.tsv`, `ingr_comp.tsv`, `srep00196-s2.csv`, `srep00196-s3.csv` | ingredient–compound data, 56k recipes |
| FooDB 2020-04-07 JSON release (foodb.ca) | `foodb_2020_04_07_json/` (project root) | nutrients |
| Recipe1MSubs (Fatemi et al. 2023, *Learning to Substitute Ingredients in Recipes*) | `data/raw/recipe1msubs/*_comments_subs.pkl` | real substitutions (ground truth) |

`data/mappings/ingredient_to_usda.csv` is the hand-curated ingredient → FooDB/USDA food mapping.

## Layout
```
src/            pipeline, models (substitution_model.py), CLI (recommend.py), evaluation
webapp/         Flask app (app.py, service.py) and the website (static/)
data/raw        source data          data/interim   FooDB extract
data/processed  matrices & tables    data/mappings  curated name mappings
results/d1..d4  figures, CSVs, *_summary.json (every number quoted in the reports)
reports/        one report per deliverable
```
