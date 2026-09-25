"""
Titanic Interactive Dashboard — Task 4
Filters, KPIs, trend charts, and a live survival predictor built on the
Task 3 model. Run with: streamlit run app.py
"""

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Titanic Dashboard", layout="wide", page_icon="🚢")


# Data / model loading (cached) 

@st.cache_data
def load_data():
    df = pd.read_csv("data/titanic_eda_features.csv")
    df["SurvivedLabel"] = df["Survived"].map({0: "Died", 1: "Survived"})
    return df


@st.cache_resource
def load_model():
    return joblib.load("model/best_model.joblib")


@st.cache_data
def load_model_comparison():
    return pd.read_csv("data/model_comparison.csv")


df = load_data()
model = load_model()
model_comparison = load_model_comparison()


# Sidebar filters 

st.sidebar.header("Filters")

pclass_sel = st.sidebar.multiselect(
    "Passenger class", sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique())
)
sex_sel = st.sidebar.multiselect(
    "Sex", sorted(df["Sex"].unique()), default=sorted(df["Sex"].unique())
)
embarked_sel = st.sidebar.multiselect(
    "Embarkation port", sorted(df["Embarked"].unique()), default=sorted(df["Embarked"].unique())
)
survived_sel = st.sidebar.multiselect(
    "Outcome", ["Survived", "Died"], default=["Survived", "Died"]
)
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Titanic (Kaggle), cleaned + feature-engineered across Tasks 1-2. "
    "Model: best pipeline from Task 3 (see Insights tab for which one and why)."
)

filtered = df[
    df["Pclass"].isin(pclass_sel)
    & df["Sex"].isin(sex_sel)
    & df["Embarked"].isin(embarked_sel)
    & df["SurvivedLabel"].isin(survived_sel)
    & df["Age"].between(age_range[0], age_range[1])
]

st.title("Titanic Dashboard")

if filtered.empty:
    st.warning("No passengers match the current filters. Widen a filter to see data.")
    st.stop()


# KPIs 

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Passengers", f"{len(filtered):,}", f"of {len(df):,} total")
k2.metric("Survival rate", f"{filtered['Survived'].mean()*100:.1f}%",
          f"{(filtered['Survived'].mean() - df['Survived'].mean())*100:+.1f} pp vs. overall")
k3.metric("Avg fare", f"${filtered['Fare'].mean():.2f}")
k4.metric("Avg age", f"{filtered['Age'].mean():.1f} yrs")
k5.metric("Avg family size", f"{filtered['FamilySize'].mean():.1f}")

st.markdown("---")


# Tabs 

tab_trends, tab_deep, tab_predict, tab_insights = st.tabs(
    ["Trends", "Deep dive", "Predict a passenger", "Insights & model"]
)

with tab_trends:
    c1, c2 = st.columns(2)

    with c1:
        surv_by_class_sex = (
            filtered.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()
        )
        fig = px.bar(
            surv_by_class_sex, x="Pclass", y="Survived", color="Sex", barmode="group",
            labels={"Survived": "Survival rate", "Pclass": "Class"},
            title="Survival rate by class & sex",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

    with c2:
        surv_by_port = filtered.groupby("Embarked")["Survived"].mean().reset_index()
        fig = px.bar(
            surv_by_port, x="Embarked", y="Survived",
            labels={"Survived": "Survival rate", "Embarked": "Port"},
            title="Survival rate by embarkation port",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2)

    with c3:
        fig = px.histogram(
            filtered, x="Age", color="SurvivedLabel", nbins=30, barmode="overlay", opacity=0.65,
            title="Age distribution by outcome",
            color_discrete_map={"Survived": "#27ae60", "Died": "#c0392b"},
        )
        st.plotly_chart(fig, width='stretch')

    with c4:
        fig = px.histogram(
            filtered, x="Fare", color="SurvivedLabel", nbins=30, barmode="overlay", opacity=0.65,
            title="Fare distribution by outcome (clipped at $200 for readability)",
            color_discrete_map={"Survived": "#27ae60", "Died": "#c0392b"},
        )
        fig.update_xaxes(range=[0, 200])
        st.plotly_chart(fig, width='stretch')

with tab_deep:
    c1, c2 = st.columns(2)

    with c1:
        fam = filtered.groupby("FamilySize")["Survived"].mean().reset_index()
        fig = px.line(
            fam, x="FamilySize", y="Survived", markers=True,
            title="Survival rate by family size (non-monotonic — see Insights)",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

    with c2:
        title_surv = filtered.groupby("Title")["Survived"].mean().reset_index()
        fig = px.bar(
            title_surv.sort_values("Survived", ascending=False), x="Title", y="Survived",
            title="Survival rate by title",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin"]
    corr = filtered[num_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation matrix (filtered data)",
    )
    st.plotly_chart(fig, width='stretch')

    st.dataframe(
        filtered[["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "SurvivedLabel"]]
        .reset_index(drop=True),
        width='stretch',
        height=300,
    )

with tab_predict:
    st.subheader("Live survival prediction")
    st.caption(
        "Runs the Task 3 model on a hypothetical passenger. This is the trained pipeline, "
        "not a lookup — it reflects the model's learned pattern, not a guaranteed outcome."
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        in_pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
        in_sex = st.selectbox("Sex", ["male", "female"])
        in_age = st.slider("Age", 0, 80, 30)
    with p2:
        in_fare = st.slider("Fare paid ($)", 0.0, 512.0, 32.0)
        in_embarked = st.selectbox("Embarkation port", ["S", "C", "Q"])
        in_family = st.slider("Family size (incl. self)", 1, 11, 1)
    with p3:
        in_title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
        in_alone = 1 if in_family == 1 else 0
        st.metric("Traveling alone?", "Yes" if in_alone else "No")

    passenger = pd.DataFrame([{
        "Pclass": in_pclass,
        "Sex": in_sex,
        "Age": in_age,
        "Fare": in_fare,
        "Embarked": in_embarked,
        "FamilySize": in_family,
        "IsAlone": in_alone,
        "Title": in_title,
    }])

    proba = model.predict_proba(passenger)[0, 1]
    pred = "Survived" if proba >= 0.5 else "Did not survive"

    st.markdown("---")
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Predicted outcome", pred, f"{proba*100:.1f}% survival probability")
    with r2:
        fig = px.bar(
            x=["Survival probability"], y=[proba], range_y=[0, 1],
            color_discrete_sequence=["#27ae60" if proba >= 0.5 else "#c0392b"],
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig, width='stretch')

with tab_insights:
    st.subheader("What the analysis found (Tasks 1-3)")
    st.markdown("""
- **Sex is the strongest single predictor** — women survived at roughly 4x the rate of men across the
  whole dataset.
- **Class and sex interact, not just add up.** A 3rd-class woman (≈50% survival) and a 1st-class man
  (≈37%) had comparable odds — sex dominates within a class, but class still swings survival 20-40
  points inside each sex. Use the Trends tab's class/sex chart to see this directly.
- **Family size is non-monotonic.** Small families (2-4 people) outperformed both solo travelers and
  large families — not a straight-line relationship.
- **Fare, Pclass, and cabin presence are collinear** — they mostly encode the same "wealth/class" signal
  from different angles, which is why the model uses `Fare` and drops the cabin flag rather than feeding
  in three redundant proxies.
- **Age alone is a weak predictor** except at the extremes (children fared noticeably better).
    """)

    st.subheader("Model comparison (Task 3, held-out test set)")
    st.dataframe(
        model_comparison.style.format({
            "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
            "f1": "{:.3f}", "roc_auc": "{:.3f}",
        }),
        width='stretch',
    )
    st.caption(
        "Logistic Regression (Ridge/L2) had the best ROC-AUC (0.874); the Stacking Ensemble had the "
        "best accuracy/F1 (0.860 / 0.812). Which one is \"best\" depends on whether ranking quality or "
        "hard classification calls matter more for the use case — see the Task 3 README for the full "
        "reasoning. The predictor above uses whichever pipeline `best_model.joblib` points to."
    )

    st.subheader("Honest limits")
    st.markdown("""
- 891 rows is small; the model's ~0.87 ROC-AUC ceiling reflects that, not a modeling shortfall.
- The predictor tab extrapolates to any input combination, including ones rare or absent in the training
  data (e.g. a very high fare in 3rd class) — treat those predictions with more skepticism than
  in-distribution ones.
- This dashboard reads a static CSV; it doesn't reflect new data unless the underlying file is refreshed
  and the app is restarted (`st.cache_data` will otherwise keep serving the old copy).
    """)
