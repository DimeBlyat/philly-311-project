# Philadelphia 311 — Abandoned Vehicle Resolution Time Analysis

Research project analyzing what factors influence how long it takes the City of
Philadelphia to resolve **Abandoned Vehicle** 311 requests, using regression,
classification, and clustering.

## Setup

```bash
pip install pandas numpy requests scikit-learn matplotlib seaborn
```

Data comes from Philadelphia's public CARTO SQL API (`phl.carto.com`, table
`public_cases_fc`). No API key registration needed — the public key
`default_public` is used in the notebook.

## Run order

1. **`01_data_acquisition.ipynb`** — connects to the CARTO API, confirms real
   table/column names, shows request-type volumes, pulls all `Abandoned Vehicle`
   rows (275,351), saves `philly311_raw.csv`.
2. **`02_cleaning.ipynb`** — parses dates, computes resolution time, sets aside
   still-open requests, drops invalid rows, engineers features, saves
   `philly311_clean.csv` (252,419 usable rows).
3. **`03_eda.ipynb`** — exploratory analysis: distribution of resolution time,
   effects of day of week, agency, zip code, and time.
4. **`04_modeling.ipynb`** — regression (predict resolution time), classification
   (fast vs. slow), and clustering.

**Note on Google Colab:** each notebook was run as a separate Colab session, so
each one re-loads its input CSV via `google.colab.files.upload()` at the top
rather than assuming a shared filesystem. If running locally instead, replace
that upload cell with a plain `pd.read_csv("path/to/file.csv")`.

## Key results

- Subset: **Abandoned Vehicle** (chosen for highest volume among the professor's
  suggested categories: 275,351 raw requests, vs. 170,339 for Street Defect,
  141,168 for Street Light Outage, 46,799 for Traffic Signal Emergency).
- Median resolution time: **~3,135 hours (~131 days / 4.3 months)**.
- Regression (Random Forest): MAE ≈ 4,696h, R² ≈ 0.06 on log scale — a real but
  weak relationship between request timing/location and resolution speed.
- Classification (fast vs. slow, median split): ~60% accuracy, meaningfully
  above the 50% baseline but far from confident.
- Clustering (k=4): clusters differ modestly in median resolution time
  (≈2,800–3,500 hours).
- `agency_responsible` and `service_name`/`service_code` were excluded as model
  features because they are constant within this subset (99.999% of Abandoned
  Vehicle requests are handled by the Police Department).

See `REPORT.md` for full methodology, problems encountered, and interpretation.

## Where an LLM was used

Project structure, starter code for data acquisition/cleaning/EDA/modeling, and
debugging help (CARTO API auth issues, a scikit-learn OneHotEncoder/dtype bug,
zip-code format inconsistencies) were done with Claude (Anthropic). All code was
run against the real dataset; every number in this repo's outputs and in
`REPORT.md` comes from an actual run, not from the LLM. Interpretation of
results and modeling decisions were reviewed and made by me (Dimitar).

## Files

| File | Purpose |
|---|---|
| `01_data_acquisition.ipynb` | Pull raw data from Philadelphia's open data API |
| `02_cleaning.ipynb` | Clean + engineer features |
| `03_eda.ipynb` | Exploratory analysis |
| `04_modeling.ipynb` | Regression, classification, clustering |
| `REPORT.md` | Written project documentation (per assignment requirements) |
| `PRESENTATION_OUTLINE.md` | Slide-by-slide outline for the presentation |
