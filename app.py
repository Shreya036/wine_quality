"""Wine Quality dashboard (Task 9, Option B: Streamlit + Plotly).

Run from the project folder with:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wine_quality_cleaned.csv")
REG_FEATURES = ["alcohol", "density", "pH", "volatile acidity", "citric acid", "residual sugar"]
LABELS = ["Not Good", "Good"]

st.set_page_config(page_title="Wine Quality Dashboard", layout="wide")


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["quality_category"] = np.where(data["quality"] >= 7, "Good", "Not Good")
    return data


@st.cache_resource
def train_everything():
    """Train all models once (same settings as notebooks 05 and 06)."""
    data = load_data()

    # ---------------- regression ----------------
    X = data[REG_FEATURES]
    y = data["quality"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    reg_rows, reg_pred = [], {}
    for name, model in reg_models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        reg_pred[name] = pred
        mse = mean_squared_error(y_te, pred)
        reg_rows.append({"Model": name, "MAE": mean_absolute_error(y_te, pred), "MSE": mse,
                         "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_te, pred)})
    reg_table = pd.DataFrame(reg_rows)
    reg_imp = permutation_importance(reg_models["Random Forest"], X_te, y_te, n_repeats=5, random_state=42,
                                     scoring="neg_root_mean_squared_error")
    reg_importance = (pd.DataFrame({"Variable": REG_FEATURES, "Importance": reg_imp.importances_mean})
                      .sort_values("Importance"))

    # ---------------- classification (quality is NOT a predictor: no target leakage) ----------------
    cls_features = [c for c in data.columns if c not in ("quality", "quality_category")]
    Xc = data[cls_features]
    yc = data["quality_category"]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    cls_models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    cls_rows, cls_cm = [], {}
    y_bin = (yc_te == "Good").astype(int)
    for name, model in cls_models.items():
        model.fit(Xc_tr, yc_tr)
        pred = model.predict(Xc_te)
        proba = model.predict_proba(Xc_te)[:, list(model.classes_).index("Good")]
        cls_cm[name] = confusion_matrix(yc_te, pred, labels=LABELS)
        cls_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yc_te, pred),
            "Precision": precision_score(yc_te, pred, pos_label="Good", zero_division=0),
            "Recall": recall_score(yc_te, pred, pos_label="Good", zero_division=0),
            "F1-Score": f1_score(yc_te, pred, pos_label="Good", zero_division=0),
            "ROC-AUC": roc_auc_score(y_bin, proba),
        })
    cls_table = pd.DataFrame(cls_rows)
    cls_imp = permutation_importance(cls_models["Random Forest"], Xc_te, yc_te, n_repeats=5, random_state=42,
                                     scoring=make_scorer(f1_score, pos_label="Good"))
    cls_importance = (pd.DataFrame({"Variable": cls_features, "Importance": cls_imp.importances_mean})
                      .sort_values("Importance"))

    return {
        "reg_models": reg_models, "reg_pred": reg_pred, "reg_table": reg_table, "y_test": y_te.to_numpy(),
        "reg_importance": reg_importance,
        "cls_models": cls_models, "cls_table": cls_table, "cls_cm": cls_cm, "cls_importance": cls_importance,
        "cls_features": cls_features,
    }


df = load_data()
res = train_everything()

st.title("Wine Quality Dashboard (white wine)")
st.caption("UCI Wine Quality data after cleaning: duplicate rows removed. "
           "Sidebar filters change the Overview tab; models are always evaluated on the full 20% test set.")

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters (Overview tab)")
q_min, q_max = int(df["quality"].min()), int(df["quality"].max())
a_min, a_max = float(df["alcohol"].min()), float(df["alcohol"].max())
quality_range = st.sidebar.slider("Quality score", min_value=q_min, max_value=q_max, value=(q_min, q_max))
alcohol_range = st.sidebar.slider("Alcohol (% by volume)", min_value=a_min, max_value=a_max, value=(a_min, a_max))

filtered = df[df["quality"].between(*quality_range) & df["alcohol"].between(*alcohol_range)]

tab_over, tab_reg, tab_cls, tab_imp, tab_find, tab_pred = st.tabs(
    ["Overview", "Regression", "Classification", "Important variables", "Findings", "Try a prediction"]
)

# ------------------------------------------------------------------ Overview
with tab_over:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total observations", f"{len(filtered):,}")
    if len(filtered) > 0:
        c2.metric("Average quality score", f"{filtered['quality'].mean():.2f}")
        c3.metric("Average alcohol (%)", f"{filtered['alcohol'].mean():.2f}")

        fig = px.histogram(filtered, x="quality", title="Distribution of quality scores",
                           labels={"quality": "Quality score", "count": "Number of wines"})
        fig.update_layout(bargap=0.1, yaxis_title="Number of wines")
        st.plotly_chart(fig)

        if len(filtered) > 10:
            corr = filtered.drop(columns="quality_category").corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Correlation of chemical properties and quality", aspect="auto")
            st.plotly_chart(fig)
        else:
            st.info("Select more wines to draw the correlation heatmap.")
    else:
        st.warning("No wines match the selected filters.")

# ------------------------------------------------------------------ Regression
with tab_reg:
    st.subheader("Regression: predicting the quality score")
    reg_table = res["reg_table"]
    st.dataframe(reg_table.round(4), hide_index=True)
    best_reg = reg_table.sort_values("RMSE").iloc[0]["Model"]
    st.success(f"Best model by lowest RMSE: {best_reg}")

    fig = px.bar(reg_table, x="Model", y="RMSE", color="Model", title="Test RMSE by model (lower is better)")
    st.plotly_chart(fig)

    chosen = st.selectbox("Model for the actual-vs-predicted plot", list(res["reg_models"].keys()),
                          index=list(res["reg_models"].keys()).index(best_reg))
    avp = pd.DataFrame({"Actual quality": res["y_test"], "Predicted quality": res["reg_pred"][chosen]})
    fig = px.scatter(avp, x="Actual quality", y="Predicted quality", opacity=0.4,
                     title=f"Actual vs predicted quality - {chosen}")
    fig.add_shape(type="line", x0=3, y0=3, x1=9, y1=9, line=dict(color="red"))
    st.plotly_chart(fig)

# ------------------------------------------------------------------ Classification
with tab_cls:
    st.subheader("Classification: Good (quality >= 7) vs Not Good")
    cls_table = res["cls_table"]
    st.dataframe(cls_table.round(4), hide_index=True)
    best_cls = cls_table.sort_values("F1-Score", ascending=False).iloc[0]["Model"]
    st.success(f"Best model by F1-score (Good class): {best_cls}")

    shown = st.selectbox("Model for the confusion matrix", list(res["cls_models"].keys()),
                         index=list(res["cls_models"].keys()).index(best_cls))
    cm = res["cls_cm"][shown]
    fig = px.imshow(cm, x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Wines"),
                    title=f"Confusion matrix - {shown}")
    st.plotly_chart(fig)
    row = cls_table[cls_table["Model"] == shown].iloc[0]
    st.metric("ROC-AUC", f"{row['ROC-AUC']:.3f}")

# ------------------------------------------------------------------ Important variables
with tab_imp:
    st.subheader("Permutation importance (Random Forest, test set)")
    left, right = st.columns(2)
    fig = px.bar(res["reg_importance"], x="Importance", y="Variable", orientation="h",
                 title="Regression: increase in RMSE when the variable is shuffled")
    left.plotly_chart(fig)
    fig = px.bar(res["cls_importance"], x="Importance", y="Variable", orientation="h",
                 title="Classification: decrease in F1 (Good) when the variable is shuffled")
    right.plotly_chart(fig)

# ------------------------------------------------------------------ Findings (computed from the data)
with tab_find:
    st.subheader("Evidence-based findings")
    corr_all = df.drop(columns="quality_category").corr()["quality"].drop("quality")
    good_alc = df.loc[df["quality"] >= 7, "alcohol"].mean()
    rest_alc = df.loc[df["quality"] < 7, "alcohol"].mean()
    reg_t = res["reg_table"].set_index("Model")
    cls_t = res["cls_table"].set_index("Model")
    baseline = (df["quality_category"] == "Not Good").mean()
    top_reg = res["reg_importance"].iloc[-1]["Variable"]   # tables are sorted ascending, last row = top
    top_cls = res["cls_importance"].iloc[-1]["Variable"]
    if top_reg == top_cls:
        top_text = f"It is also the most important variable in both Random Forest models ({top_reg})."
    else:
        top_text = f"The top Random Forest variable is {top_reg} for regression and {top_cls} for classification."

    st.markdown(
        f"1. **Alcohol is the variable most linked to quality** (correlation {corr_all['alcohol']:+.2f}); wines rated 7 or "
        f"above average {good_alc:.2f}% alcohol versus {rest_alc:.2f}% for the rest. {top_text}\n"
        f"2. **Chemistry explains only part of quality.** The best regressor ({best_reg}) reaches R2 = "
        f"{reg_t.loc[best_reg, 'R2']:.2f} and RMSE = {reg_t.loc[best_reg, 'RMSE']:.2f} quality points, "
        f"barely better than Linear Regression (R2 = {reg_t.loc['Linear Regression', 'R2']:.2f}); the unrestricted "
        f"Decision Tree overfits (R2 = {reg_t.loc['Decision Tree', 'R2']:.2f}).\n"
        f"3. **Good wines are hard to find.** Only {1 - baseline:.1%} of wines are Good. The {best_cls} reaches "
        f"{cls_t.loc[best_cls, 'Accuracy']:.1%} accuracy (always answering Not Good gives {baseline:.1%}) with "
        f"recall of {cls_t.loc[best_cls, 'Recall']:.0%} and ROC-AUC {cls_t.loc[best_cls, 'ROC-AUC']:.2f}, so accuracy "
        f"alone overstates performance."
    )
    st.caption("Association, not causation: these results describe patterns in the data only.")

# ------------------------------------------------------------------ Live prediction
with tab_pred:
    st.subheader("Predict the quality of a wine")
    st.caption("Set the chemical measurements; the Random Forest models return a quality score and the "
               "probability that the wine is Good.")
    values = {}
    cols = st.columns(3)
    for i, feat in enumerate(res["cls_features"]):
        with cols[i % 3]:
            values[feat] = st.slider(feat, min_value=float(df[feat].min()), max_value=float(df[feat].max()),
                                     value=float(df[feat].median()), key=f"input_{feat}")

    reg_input = pd.DataFrame([{f: values[f] for f in REG_FEATURES}])
    cls_input = pd.DataFrame([values])[res["cls_features"]]
    rf_cls = res["cls_models"]["Random Forest"]
    p_good = rf_cls.predict_proba(cls_input)[0, list(rf_cls.classes_).index("Good")]

    m1, m2 = st.columns(2)
    m1.metric("Predicted quality score", f"{res['reg_models']['Random Forest'].predict(reg_input)[0]:.2f}")
    m2.metric("Probability the wine is Good", f"{p_good:.0%}")
