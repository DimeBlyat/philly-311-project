# Philadelphia 311: Factors Affecting Abandoned Vehicle Resolution Time

*[Your name] — [Course] — [Date]*

> This draft is filled in with real results from the actual notebook runs. Replace
> `[your name]`/`[date]`/`[course]` above, review every section for accuracy against
> your own final notebook outputs, and add the two items marked **TODO** below before
> submitting.

## 1. Research Question

How does resolution time for Philadelphia 311 **Abandoned Vehicle** requests relate
to when and where a request is filed? Can resolution time be predicted, and do
requests fall into meaningfully different groups based on their profile?

## 2. Data

- **Source:** City of Philadelphia open data, 311 Service and Information Requests
  (CARTO SQL API, `phl.carto.com`, table `public_cases_fc`).
- **Subset:** Abandoned Vehicle, chosen because it had the highest volume among the
  four categories suggested for this project (275,351 requests, vs. 170,339 for
  Street Defect, 141,168 for Street Light Outage, and 46,799 for Traffic Signal
  Emergency — the latter's low volume and "emergency" framing suggested less
  variability worth modeling).
- **Size:** 275,351 raw rows pulled; 252,419 rows remained after cleaning.
- **Key variables used:** `zipcode`, `day_of_week`, `hour_of_day`, `month`,
  `is_weekend` (derived from `requested_datetime`), target `resolution_hours`
  (= `closed_datetime` − `requested_datetime`).

## 3. Preprocessing

- Started with 275,351 raw rows for the Abandoned Vehicle subset.
- 0 rows had a missing created-date timestamp.
- **22,932 requests (8.3%)** had no closed date and were set aside as a separate
  "still open" dataset rather than dropped, since they're meaningful context (a
  substantial share of requests remain unresolved) even though they can't be used
  as labeled examples for regression/classification.
- 0 rows had negative resolution time (no data-entry errors of that kind found in
  this subset).
- 0 duplicate request IDs.
- Missingness in remaining fields was low: zipcode 0.6%, lat/lon 0.2% each,
  agency_responsible 0.0%.
- Resulting clean dataset: **252,419 rows**.
- Applied `log1p` to resolution time for modeling, since the raw distribution is
  heavily right-skewed (see EDA below).
- For modeling (notebook 04), rows missing any of the five chosen features were
  additionally dropped, leaving 250,898 rows. This is why the classification
  threshold computed in notebook 04 (3,158.7h) differs very slightly from the
  median computed on the full clean dataset in notebook 03 (3,134.7h) — both are
  reported here for transparency; the difference is negligible (<1 day).
- `service_name`, `service_code`, and `agency_responsible` were **excluded as
  model features**: because only one request type was selected, `service_name`/
  `service_code` are constant by construction, and `agency_responsible` is
  effectively constant too (252,416 of 252,419 rows — 99.999% — are handled by
  the Police Department). Constant columns carry no predictive information and
  would only add noise.

## 4. Problems Encountered

- **CARTO API access:** initial queries against `information_schema` (used to
  discover the real table/column names) returned `403 Forbidden`, even though the
  data table itself was public. Direct queries against the table also failed with
  `403` until the public API key parameter `api_key=default_public` was added to
  every request — CARTO requires this even for nominally public datasets.
- **Zip code format inconsistency:** most `zipcode` values were plain 5-digit
  numbers (e.g. `19104.0`), but at least one record used ZIP+4 format
  (`19101-0796`), which broke a naive `.astype(int)` conversion. Fixed by
  splitting on `.`/`-` and keeping only the leading 5 digits.
- **scikit-learn encoding bug:** `OneHotEncoder(min_frequency=20)` raised a
  `TypeError` when the zip code column mixed floats and its internal
  `"infrequent_sklearn"` string label during sorting. Fixed by converting zip
  codes to clean strings before fitting.
- **A long-tail anomaly in resolution time:** a small cluster of requests
  (n=463, 0.18% of the clean data) had resolution times of ~36,000–46,000 hours
  (4–5 years). Checking `status_notes` for these rows showed 426 of 463 were
  closed with **"Information Provided"** rather than "Issue Resolved" — meaning
  these are very likely administrative closures (someone responding to an old,
  stale ticket) rather than genuine vehicle removals. This was left in the
  dataset (0.18% is too small to meaningfully bias the results) but is
  worth noting as a data-quality caveat rather than treating those durations as
  literal "how long resolution took."
- **Volume-over-time chart artifact:** the monthly request-volume chart shows an
  apparent decline toward the most recent months in the dataset. This is not a
  real drop in reporting — it's an artifact of excluding still-open requests:
  recent requests haven't had time to close yet, so recent months are
  under-represented in a "closed requests per month" count.

## 5. Methods

### Regression
- Target: `log1p(resolution_hours)`.
- Models: baseline (predict median), linear regression, random forest
  (200 trees, max depth 12).
- Rationale: linear regression as an interpretable baseline; random forest to
  capture potential non-linear interactions between zip code, time, and weekday.

### Classification
- Target: binary fast/slow, split at the median resolution time
  (**3,158.7 hours ≈ 131.6 days**), giving an exactly balanced 50/50 class split.
- Models: baseline (majority class), logistic regression, random forest.
- Metrics: accuracy, precision, recall, and F1 rather than accuracy alone —
  chosen because with only mildly informative features, a model could achieve
  deceptively "OK" accuracy while being systematically biased toward one class;
  precision/recall exposes that.

### Clustering
- Features: `zipcode`, `day_of_week`, `hour_of_day`, `month`, `is_weekend`
  (`resolution_hours` itself was excluded from the clustering features, so that
  clusters describe request *profile* rather than trivially splitting on the
  outcome we're trying to explain).
- Method: k-means, k chosen by inspecting elbow and silhouette plots over
  k = 2–8; **k = 4** was used for the final clustering.

## 6. Evaluation and Results

**Regression** (metrics on original hours scale; R² measured on the log scale
the models were actually fit on):

| Model | MAE (hours) | RMSE (hours) | R² (log scale) |
|---|---|---|---|
| Baseline (median) | 4,789.3 | 8,168.3 | -0.001 |
| Linear Regression | 4,675.5 | 8,083.5 | 0.046 |
| Random Forest | 4,695.9 | 8,095.0 | 0.060 |

**Classification** (fast vs. slow, median split):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (majority class) | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.597 | 0.595 | 0.608 | 0.602 |
| Random Forest | 0.597 | 0.589 | 0.639 | 0.613 |

Random Forest confusion matrix (test set, n=50,180):
- True fast, predicted fast: 13,929
- True fast, predicted slow: 11,161
- True slow, predicted fast: 9,070
- True slow, predicted slow: 16,020

**Clustering** (k=4):

| Cluster | n | Median resolution (hours) |
|---|---|---|
| 0 | 17,420 | 2,893.9 |
| 1 | 108,712 | 2,803.5 |
| 2 | 100,112 | 3,503.8 |
| 3 | 24,654 | 3,421.7 |

A follow-up breakdown by zip, hour, and weekend share (since type/agency are
constant and uninformative) clarifies what actually distinguishes the clusters:

| Cluster | n | Median resolution (h) | Top zip | Avg. hour filed | % weekend |
|---|---|---|---|---|---|
| 0 | 17,420 | 2,893.9 | 19143 | ~2:40 AM | 0% |
| 1 | 108,712 | 2,803.5 | 19124 | ~4:48 PM | 0% |
| 2 | 100,112 | 3,503.8 | 19124 | ~4:39 PM | 0% |
| 3 | 24,654 | 3,421.7 | 19120 | ~2:27 PM | 100% |

**Plain-language cluster descriptions:**
- **Cluster 0** — a smaller group (7% of data) of weekday requests filed in the
  very early morning, concentrated in zip 19143. Resolves relatively quickly
  (median ~121 days).
- **Cluster 1** — the largest cluster (43% of data): weekday, late-afternoon
  requests concentrated in zip 19124. The fastest-resolving cluster overall
  (median ~117 days).
- **Cluster 2** — nearly identical profile to Cluster 1 (same top zip, same
  time of day, all weekday), but resolves noticeably slower (median ~146 days,
  about a month longer). Since location and time-of-day look the same as
  Cluster 1, the actual driver separating these two is most likely request
  month/season, which isn't captured in this summary table.
- **Cluster 3** — the only exclusively-weekend cluster (100%), concentrated in
  zip 19120, filed in early-to-mid afternoon. Resolves somewhat slower than
  weekday clusters on average (median ~143 days).

## 7. Interpretation

- **Both regression and classification show a real but weak relationship**
  between *when/where* a request is filed and how quickly it's resolved. An R² of
  0.06 means the five features used explain only about 6% of the variation in
  resolution time; classification reaches ~60% accuracy, meaningfully above the
  50% coin-flip baseline but far from a confident prediction.
- This suggests resolution time for Abandoned Vehicle requests is driven mainly
  by **factors not captured in this dataset** — most plausibly operational
  capacity (how backed up towing/police response is at a given time), legal
  holding periods before a vehicle can be towed, or case-specific circumstances —
  rather than by the request's timing or location per se.
- The Random Forest classifier is mildly biased toward predicting "slow"
  (catches 63.9% of genuinely slow cases vs. 55.5% of genuinely fast ones). In a
  practical triage setting, that bias might be acceptable or even desirable — a
  tool that over-flags potential delays costs less than one that misses them.
- Feature importance from the Random Forest ranked `hour_of_day` and `month` far
  above zip code or weekday — but this should be read cautiously: scikit-learn's
  default importance measure is known to be biased toward high-cardinality/
  continuous features over one-hot categorical ones, and this ranking is at odds
  with the EDA finding that day-of-week had almost no visible relationship with
  resolution time. A permutation-importance check would give a more trustworthy
  ranking (see notebook 04 for the optional code to run this).
- The clusters found by k-means differ by roughly 600–700 hours in median
  resolution time between the fastest and slowest groups — a real but modest
  split, consistent with the weak regression/classification signal: request
  *profile* alone doesn't sort requests into dramatically different outcome
  groups.
- The dominant axis k-means found was **weekend vs. weekday** (Cluster 3 is
  100% weekend, the other three are 0%) — a clean categorical separation.
  Importantly, this is a case where "the clusters look very different" does
  *not* imply "weekend filing predicts resolution speed": the EDA in notebook
  03 already showed day-of-week has almost no visible relationship with
  resolution time. K-means separates on whichever features create the
  most geometrically distinct groups, not necessarily the features that most
  affect the outcome — a useful reminder that clustering answers "how do
  requests group by profile?", not "what causes fast vs. slow resolution?".
  The two largest clusters (1 and 2) share the same top zip code and time of
  day yet differ by about a month in typical resolution time, suggesting
  seasonal/monthly timing — not location or time-of-day — is the more likely
  driver of that particular gap.

## 8. Conclusion

For Abandoned Vehicle 311 requests in Philadelphia, resolution time is
**long** (median ~131 days) and only weakly explained by when or where a request
is filed. Location (zip code) carries more signal than timing (day of week, in
particular, carries almost none). This points toward resolution time being
primarily a function of operational capacity and case-specific factors external
to this dataset, rather than of citizen-controllable factors like when they
report an issue. A natural next step, with more data, would be to incorporate
operational variables — towing queue length, staffing levels, or whether a legal
hold period applies — which this open dataset does not include.

**Limitations:** the exclusion of still-open requests (8.3% of the data) means
recent months are under-represented in any time-based analysis, and a small
subset of very-long-duration closures (0.18%) may represent administrative
closures rather than genuine resolutions. Both are documented above but not
fully corrected for.

## 9. Use of LLM Assistance

Claude (Anthropic) was used to scaffold the data-pull, cleaning, EDA, and
modeling code across all four notebooks, and to help debug several issues
(CARTO API authentication, a scikit-learn encoding bug, zip code format
inconsistencies). All code was run against the real dataset by [your name]; every
number in this report comes from an actual notebook run. Interpretation of
results, the choice of subset and modeling decisions, and this report's
conclusions were reviewed and made by [your name].
