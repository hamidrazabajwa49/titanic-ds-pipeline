# Task 4 — Interactive Data Dashboard: Titanic

| | |
|---|---|
| Tool | Streamlit + Plotly |
| Data | `titanic_eda_features.csv` (Task 2 output) |
| Model | `best_model.joblib` (Task 3 output — Logistic Regression, best test ROC-AUC) |
| Filters | Class, sex, embarkation port, outcome, age range |
| KPIs | Passenger count, survival rate (vs. overall delta), avg fare, avg age, avg family size |
| Tabs | Trends · Deep dive · Predict a passenger · Insights & model |
| Tested | `streamlit.testing.v1.AppTest` — 0 exceptions across default load, filter changes, predictor input changes, and the empty-filter edge case |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## What's in it

- **Trends tab** — survival rate by class & sex, by embarkation port, age and fare distributions split
  by outcome. All charts respect the sidebar filters.
- **Deep dive tab** — family size vs. survival (the non-monotonic pattern from Task 2), survival by
  title, a live correlation matrix on the filtered data, and a raw data table.
- **Predict a passenger tab** — set class, sex, age, fare, port, family size, and title; runs the actual
  Task 3 model and returns a survival probability, not a lookup.
- **Insights & model tab** — the headline findings from Tasks 1-3 in plain language, the Task 3 model
  comparison table, and an honest-limits section (small dataset, extrapolation risk on the predictor,
  static data source).

## Honest Take

- Tested with `AppTest`, not just "it imports" — actually ran the script, changed filters, changed
  predictor inputs, and hit the empty-filter path, all with 0 exceptions. Confirmed the server also
  boots and serves on port 8501.
- The predictor extrapolates to combinations barely or never seen in training (e.g. a $500 fare in
  3rd class) — the app doesn't hide that, it says so directly in Insights.
- KPIs and charts all read off the same filtered dataframe, so nothing gets out of sync between a
  filter change and what's displayed.
- Data is a static CSV loaded once and cached (`st.cache_data`) — refreshing the underlying file
  requires an app restart, not just a page reload. Noted in-app, not left as a silent gotcha.

## Project structure

```
Task 4/
├── data/
│   └── titanic_eda_features.csv    # data, from Task 2
├── model/
│   ├── best_model.joblib           # model, from Task 3
│   └── model_comparison.csv        # from Task 3, shown in Insights tab
├── app.py                          # Streamlit app
├── requirements.txt
└── README.md
```

> **Path note:** `app.py` currently loads `titanic_eda_features.csv`, `best_model.joblib`, and
> `model_comparison.csv` from its own working directory. Since the data now lives in `data/` and the
> model in `model/`, those three `pd.read_csv(...)` / `joblib.load(...)` calls need their paths updated
> (e.g. `data/titanic_eda_features.csv`, `model/best_model.joblib`) or the app won't find the files when
> run from this layout. Say the word and I'll make that fix.
