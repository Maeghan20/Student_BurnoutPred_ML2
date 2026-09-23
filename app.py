import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Burnout Lab | Regularized Regression",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# A small CSS layer makes the app feel like a polished dashboard
# instead of a default Streamlit page.
st.markdown("""
<style>
    .stApp { background: #0b1020; }
    [data-testid="stSidebar"] { background: #11182d; }
    .hero {
        padding: 2.2rem 2.4rem;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 22px;
        background: linear-gradient(135deg, #151e3d 0%, #10172b 55%, #162a35 100%);
        margin-bottom: 1.4rem;
    }
    .hero h1 { margin: 0; font-size: 2.35rem; }
    .hero p { color: #aeb9d4; margin: .55rem 0 0; font-size: 1.03rem; }
    .tag {
        display: inline-block;
        padding: .32rem .72rem;
        border-radius: 999px;
        background: rgba(120, 170, 255, .12);
        border: 1px solid rgba(120, 170, 255, .25);
        color: #b9d3ff;
        font-size: .78rem;
        margin-right: .35rem;
    }
    .section-card {
        padding: 1.1rem 1.25rem;
        border-radius: 17px;
        background: #11182d;
        border: 1px solid rgba(255,255,255,.08);
        margin-bottom: 1rem;
    }
    .small-muted { color: #9ca8c1; font-size: .9rem; }
    div[data-testid="stMetric"] {
        background: #11182d;
        border: 1px solid rgba(255,255,255,.08);
        padding: 1rem;
        border-radius: 15px;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA + MODEL HELPERS
# ============================================================
@st.cache_data
def load_data():
    # The CSV is kept inside the same repository as this app,
    # so Streamlit Cloud can load it without any local path.
    return pd.read_csv("student_burnout.csv")


def make_preprocessor(X):
    """Build preprocessing for mixed numeric + categorical data.

    Numeric columns:
      - median imputation handles missing values
      - StandardScaler puts variables on a comparable scale

    Categorical columns:
      - most-frequent imputation handles missing categories
      - OneHotEncoder converts text into model-ready numbers
    """
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical)
    ])


def metric_dict(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R²": r2_score(y_true, y_pred)
    }


@st.cache_resource
def train_all_models(random_state=42):
    """Train the required models using one fixed train/test split.

    Linear Regression is the baseline.
    Lasso and Ridge use GridSearchCV with 5-fold CV to find alpha.
    A Polynomial Ridge model is an optional self-learning extension.
    """
    df = load_data()

    # high_burnout is derived from burnout_score, so using it would
    # leak target information into the model. student_id is an identifier.
    X = df.drop(columns=["burnout_score", "high_burnout", "student_id"])
    y = df["burnout_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    preprocessor = make_preprocessor(X)

    # 1) Required baseline
    linear = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
    linear.fit(X_train, y_train)
    pred_linear = linear.predict(X_test)

    # A logarithmic grid gives enough alpha values to explore
    # very weak to fairly strong regularization.
    alpha_grid = np.logspace(-4, 2, 30)

    # 2) Required Lasso + 5-fold GridSearchCV
    lasso_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Lasso(max_iter=20000))
    ])
    lasso_grid = GridSearchCV(
        lasso_pipe,
        param_grid={"model__alpha": alpha_grid},
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    lasso_grid.fit(X_train, y_train)
    pred_lasso = lasso_grid.predict(X_test)

    # 3) Required Ridge + 5-fold GridSearchCV
    ridge_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ])
    ridge_grid = GridSearchCV(
        ridge_pipe,
        param_grid={"model__alpha": alpha_grid},
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    ridge_grid.fit(X_train, y_train)
    pred_ridge = ridge_grid.predict(X_test)

    # 4) Self-learning extension:
    # Polynomial interactions + Ridge can model simple interactions
    # while Ridge controls coefficient size.
    poly_preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(
                degree=2, include_bias=False, interaction_only=True
            ))
        ]), X.select_dtypes(exclude=["object", "category"]).columns.tolist()),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ]), X.select_dtypes(include=["object", "category"]).columns.tolist())
    ])

    poly_ridge_grid = GridSearchCV(
        Pipeline([
            ("preprocessor", poly_preprocessor),
            ("model", Ridge())
        ]),
        param_grid={"model__alpha": np.logspace(-2, 3, 25)},
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    poly_ridge_grid.fit(X_train, y_train)
    pred_poly = poly_ridge_grid.predict(X_test)

    predictions = {
        "Linear Regression": pred_linear,
        "Lasso": pred_lasso,
        "Ridge": pred_ridge,
        "Polynomial Ridge (self-learning)": pred_poly
    }

    results = pd.DataFrame(
        {name: metric_dict(y_test, pred) for name, pred in predictions.items()}
    ).T

    # Extract readable coefficients from the fitted preprocessing pipeline.
    # This is useful for the required coefficient comparison.
    def get_coefficients(fitted_pipeline):
        prep = fitted_pipeline.named_steps["preprocessor"]
        model = fitted_pipeline.named_steps["model"]
        names = prep.get_feature_names_out()
        return pd.Series(model.coef_, index=names)

    coef_linear = get_coefficients(linear)
    coef_lasso = get_coefficients(lasso_grid.best_estimator_)
    coef_ridge = get_coefficients(ridge_grid.best_estimator_)

    # GridSearchCV stores mean CV scores for every alpha.
    lasso_cv = pd.DataFrame({
        "alpha": lasso_grid.cv_results_["param_model__alpha"].astype(float),
        "RMSE": -lasso_grid.cv_results_["mean_test_score"]
    }).sort_values("alpha")

    ridge_cv = pd.DataFrame({
        "alpha": ridge_grid.cv_results_["param_model__alpha"].astype(float),
        "RMSE": -ridge_grid.cv_results_["mean_test_score"]
    }).sort_values("alpha")

    return {
        "df": df,
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "linear": linear,
        "lasso": lasso_grid.best_estimator_,
        "ridge": ridge_grid.best_estimator_,
        "poly_ridge": poly_ridge_grid.best_estimator_,
        "predictions": predictions,
        "results": results,
        "lasso_alpha": float(lasso_grid.best_params_["model__alpha"]),
        "ridge_alpha": float(ridge_grid.best_params_["model__alpha"]),
        "poly_alpha": float(poly_ridge_grid.best_params_["model__alpha"]),
        "coef_linear": coef_linear,
        "coef_lasso": coef_lasso,
        "coef_ridge": coef_ridge,
        "lasso_cv": lasso_cv,
        "ridge_cv": ridge_cv
    }


# ============================================================
# LOAD / TRAIN
# ============================================================
df = load_data()
bundle = train_all_models()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## ◈ BURNOUT LAB")
st.sidebar.caption("MAI511-2 • Regularized Regression")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Explore Data",
        "Preprocessing",
        "Model Comparison",
        "Alpha & Coefficients",
        "Self-Learning",
        "Predict Burnout"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Dataset")
st.sidebar.write("2,000 records • 17 columns")
st.sidebar.caption("Target")
st.sidebar.write("burnout_score")
st.sidebar.caption("Split")
st.sidebar.write("80% train / 20% test • random_state=42")

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div>
      <span class="tag">REGRESSION</span>
      <span class="tag">LASSO</span>
      <span class="tag">RIDGE</span>
      <span class="tag">GRID SEARCH</span>
    </div>
    <h1>Student Burnout Intelligence</h1>
    <p>Understanding how regularization changes a regression model's errors, coefficients, and stability.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.subheader("Lab objective")
    st.write(
        "Build a Linear Regression baseline, then use Lasso and Ridge regularization "
        "with 5-fold GridSearchCV to find the best penalty strength (alpha). "
        "The experiment compares prediction error, R², coefficient behavior, and "
        "the effect of feature scaling."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{df.isna().sum().sum():,}")
    c4.metric("Duplicate rows", f"{df.duplicated().sum():,}")

    st.markdown("### What is being predicted?")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("""
        **Target:** `burnout_score`  
        **Target type:** continuous numeric score (1–5)

        We intentionally remove:
        - `student_id` → identifier, not a meaningful predictor
        - `high_burnout` → derived from the burnout outcome and would cause target leakage

        `gender` is categorical and is one-hot encoded.
        Missing numeric values are median-imputed and missing categorical values
        are filled with the most frequent category.
        """)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.histplot(df["burnout_score"], discrete=True, ax=ax)
        ax.set_title("Distribution of burnout_score")
        ax.set_xlabel("Burnout score")
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Required experiment")
    st.info(
        "Linear Regression → Lasso + 5-fold GridSearchCV → Ridge + 5-fold GridSearchCV "
        "→ compare MAE, MSE, RMSE and R² → inspect coefficients → interpret alpha."
    )

# ============================================================
# PAGE 2: DATA EXPLORATION
# ============================================================
elif page == "Explore Data":
    st.subheader("Explore the dataset")

    a, b, c = st.columns(3)
    a.metric("Shape", f"{df.shape[0]} × {df.shape[1]}")
    b.metric("Missing values", f"{df.isna().sum().sum()}")
    c.metric("Duplicates", f"{df.duplicated().sum()}")

    tab1, tab2, tab3 = st.tabs(["Preview", "Descriptive statistics", "Relationships"])

    with tab1:
        st.dataframe(df.head(15), use_container_width=True)
        st.caption("The dataset contains student lifestyle, academic, support, stress and burnout variables.")

    with tab2:
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab3:
        numeric_features = [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in ["student_id", "burnout_score", "high_burnout"]
        ]
        selected = st.selectbox("Choose a feature", numeric_features, index=0)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.regplot(
            data=df, x=selected, y="burnout_score",
            scatter_kws={"alpha": 0.35}, line_kws={"linewidth": 2}, ax=ax
        )
        ax.set_title(f"{selected} vs burnout_score")
        st.pyplot(fig, clear_figure=True)

        corr = df.select_dtypes(include=np.number).corr()["burnout_score"].drop(
            ["burnout_score", "student_id", "high_burnout"]
        ).sort_values()
        fig2, ax2 = plt.subplots(figsize=(9, 4.8))
        corr.plot(kind="barh", ax=ax2)
        ax2.set_title("Numeric feature correlation with burnout_score")
        ax2.set_xlabel("Pearson correlation")
        st.pyplot(fig2, clear_figure=True)

# ============================================================
# PAGE 3: PREPROCESSING
# ============================================================
elif page == "Preprocessing":
    st.subheader("Data preprocessing pipeline")

    st.markdown("""
    ### Why each step is needed

    **1. Remove leakage / identifiers**  
    `student_id` is an identifier. `high_burnout` is derived from the target, so
    keeping it would let the model indirectly see the answer.

    **2. Handle missing values**  
    The dataset contains missing values in `screen_time_hours`,
    `commute_minutes`, and `teacher_support`. Median imputation is used for
    numeric variables; most-frequent imputation is used for categorical variables.

    **3. Encode categorical data**  
    `gender` contains text labels, so One-Hot Encoding converts it to numeric
    indicator columns.

    **4. Scale numeric variables**  
    StandardScaler gives numeric features a comparable scale. This is especially
    important for Lasso and Ridge because their penalties depend on coefficient
    magnitude.

    **5. Use a Pipeline**  
    The preprocessing is fitted only on the training data inside the pipeline,
    reducing the risk of data leakage.
    """)

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        st.markdown("### Missing-value profile")
        st.dataframe(missing.rename("missing_count").to_frame(), use_container_width=True)

    st.markdown("### Train/test setup")
    st.code(
        "train_test_split(X, y, test_size=0.20, random_state=42)",
        language="python"
    )
    st.write(
        f"Training rows: **{len(bundle['X_train']):,}**  |  "
        f"Testing rows: **{len(bundle['X_test']):,}**"
    )

# ============================================================
# PAGE 4: MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.subheader("Model performance")

    display = bundle["results"].copy()
    display["R²"] = display["R²"].map(lambda x: f"{x:.4f}")
    display["MAE"] = display["MAE"].map(lambda x: f"{x:.4f}")
    display["MSE"] = display["MSE"].map(lambda x: f"{x:.4f}")
    display["RMSE"] = display["RMSE"].map(lambda x: f"{x:.4f}")

    st.dataframe(display, use_container_width=True)

    st.caption(
        "For MAE/MSE/RMSE, lower is better. For R², higher is better. "
        "R² is not classification accuracy; it is the proportion of target variance explained."
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bundle["results"]["R²"].plot(kind="bar", ax=ax)
    ax.set_ylabel("R²")
    ax.set_ylim(0, min(1.0, max(bundle["results"]["R²"].max() + 0.12, 0.7)))
    ax.set_title("R² comparison")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig, clear_figure=True)

    selected_model = st.selectbox(
        "Inspect predictions",
        ["Lasso", "Ridge"],
        index=0
    )
    pred = bundle["predictions"][selected_model]
    actual = bundle["y_test"]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(actual, pred, alpha=0.55)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    ax2.plot([lo, hi], [lo, hi], linestyle="--")
    ax2.set_xlabel("Actual burnout score")
    ax2.set_ylabel("Predicted burnout score")
    ax2.set_title(f"Actual vs Predicted — {selected_model}")
    st.pyplot(fig2, clear_figure=True)

    residuals = actual - pred
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.scatter(pred, residuals, alpha=0.55)
    ax3.axhline(0, linestyle="--")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Residual (actual - predicted)")
    ax3.set_title(f"Residual plot — {selected_model}")
    st.pyplot(fig3, clear_figure=True)

# ============================================================
# PAGE 5: ALPHA + COEFFICIENTS
# ============================================================
elif page == "Alpha & Coefficients":
    st.subheader("Regularization: alpha and coefficients")

    a, b = st.columns(2)
    a.metric("Best Lasso α", f"{bundle['lasso_alpha']:.6g}")
    b.metric("Best Ridge α", f"{bundle['ridge_alpha']:.6g}")

    st.markdown("""
    **Interpretation:** alpha controls the strength of regularization.
    A very small alpha behaves closer to ordinary Linear Regression.
    Increasing alpha makes the penalty stronger, shrinking coefficients toward zero.
    Lasso can make some coefficients exactly zero; Ridge usually keeps coefficients
    non-zero but smaller.
    """)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(bundle["lasso_cv"]["alpha"], bundle["lasso_cv"]["RMSE"], marker="o", markersize=3)
    ax.axvline(bundle["lasso_alpha"], linestyle="--", label=f"Best α = {bundle['lasso_alpha']:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("5-fold CV RMSE")
    ax.set_title("Lasso cross-validation performance")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(9, 4.8))
    ax2.plot(bundle["ridge_cv"]["alpha"], bundle["ridge_cv"]["RMSE"], marker="o", markersize=3)
    ax2.axvline(bundle["ridge_alpha"], linestyle="--", label=f"Best α = {bundle['ridge_alpha']:.4g}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Alpha (log scale)")
    ax2.set_ylabel("5-fold CV RMSE")
    ax2.set_title("Ridge cross-validation performance")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

    coef = pd.DataFrame({
        "Linear Regression": bundle["coef_linear"],
        "Lasso": bundle["coef_lasso"],
        "Ridge": bundle["coef_ridge"]
    }).fillna(0)

    coef["Lasso absolute coefficient"] = coef["Lasso"].abs()
    coef = coef.sort_values("Lasso absolute coefficient", ascending=False).drop(
        columns="Lasso absolute coefficient"
    )

    st.markdown("### Coefficient comparison")
    st.dataframe(coef, use_container_width=True)

    zero_features = coef.index[np.isclose(coef["Lasso"], 0, atol=1e-8)].tolist()
    st.markdown("### Features set to zero by Lasso")
    if zero_features:
        st.write(", ".join(zero_features))
    else:
        st.info("For this split and alpha, Lasso did not drive any displayed coefficient exactly to zero.")

# ============================================================
# PAGE 6: SELF LEARNING
# ============================================================
elif page == "Self-Learning":
    st.subheader("Self-learning & additional exploration")

    st.success(
        "Extension explored: Polynomial interaction features + Ridge regularization. "
        "This tests whether simple feature interactions can reduce error beyond the required models."
    )

    base_results = bundle["results"].loc[["Linear Regression", "Lasso", "Ridge"]]
    ext = bundle["results"].loc["Polynomial Ridge (self-learning)"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Polynomial Ridge α", f"{bundle['poly_alpha']:.5g}")
    c2.metric("Extension RMSE", f"{ext['RMSE']:.4f}")
    c3.metric("Extension R²", f"{ext['R²']:.4f}")

    st.markdown("""
    ### What I learned from the extension

    **PolynomialFeatures (degree 2, interaction_only=True)** creates pairwise
    interactions such as `sleep_hours × homework_hours` without creating squared
    terms. This gives the model a chance to represent relationships that are not
    purely additive.

    **Why Ridge is used after expansion:** interaction features increase the
    number of predictors and can introduce correlated features. Ridge shrinks
    coefficients and helps stabilize the expanded model.

    The extension is not used to replace the required Lasso/Ridge experiment.
    It is an additional investigation focused on reducing prediction error.
    """)

    st.markdown("### Error reduction relative to baseline")
    for name in ["Lasso", "Ridge", "Polynomial Ridge (self-learning)"]:
        rmse_change = base_results.loc["Linear Regression", "RMSE"] - bundle["results"].loc[name, "RMSE"]
        r2_change = bundle["results"].loc[name, "R²"] - base_results.loc["Linear Regression", "R²"]
        st.write(
            f"**{name}:** RMSE change = **{rmse_change:+.4f}**, "
            f"R² change = **{r2_change:+.4f}**"
        )

    st.markdown("### Responsible interpretation")
    st.warning(
        "A lower test error on this fixed split does not prove that a model will "
        "generalize better to every student population. Cross-validation is useful "
        "for model selection, while the held-out test set gives the final comparison."
    )

# ============================================================
# PAGE 7: PREDICTION
# ============================================================
elif page == "Predict Burnout":
    st.subheader("Try a new student profile")
    st.caption("Enter a profile and let the selected trained model estimate the burnout score.")

    model_name = st.selectbox("Prediction model", ["Lasso", "Ridge", "Linear Regression"])

    col1, col2, col3 = st.columns(3)
    with col1:
        grade = st.number_input("Grade", min_value=9, max_value=12, value=10, step=1)
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique().tolist()))
        sleep_hours = st.number_input("Sleep hours", 3.5, 11.0, 7.5, 0.1)
        sleep_quality = st.slider("Sleep quality", 1, 5, 4)
        homework_hours = st.number_input("Homework hours", 0.0, 6.4, 2.5, 0.1)

    with col2:
        tests_per_week = st.number_input("Tests per week", 0, 8, 2)
        extracurricular_hours = st.number_input("Extracurricular hours", 0.0, 18.8, 6.0, 0.1)
        num_activities = st.number_input("Number of activities", 0, 7, 2)
        screen_time_hours = st.number_input("Screen time hours", 0.5, 9.8, 4.0, 0.1)
        commute_minutes = st.number_input("Commute minutes", 2.0, 77.0, 23.0, 1.0)

    with col3:
        family_support = st.slider("Family support", 1, 5, 4)
        friend_support = st.slider("Friend support", 1, 5, 4)
        teacher_support = st.slider("Teacher support", 1.0, 5.0, 3.0, 0.5)
        self_rated_stress = st.slider("Self-rated stress", 1, 5, 3)

    if st.button("Estimate burnout", type="primary", use_container_width=True):
        new_student = pd.DataFrame([{
            "grade": grade,
            "gender": gender,
            "sleep_hours": sleep_hours,
            "sleep_quality": sleep_quality,
            "homework_hours": homework_hours,
            "tests_per_week": tests_per_week,
            "extracurricular_hours": extracurricular_hours,
            "num_activities": num_activities,
            "screen_time_hours": screen_time_hours,
            "commute_minutes": commute_minutes,
            "family_support": family_support,
            "friend_support": friend_support,
            "teacher_support": teacher_support,
            "self_rated_stress": self_rated_stress
        }])

        model_map = {
            "Linear Regression": bundle["linear"],
            "Lasso": bundle["lasso"],
            "Ridge": bundle["ridge"]
        }
        prediction = float(model_map[model_name].predict(new_student)[0])
        prediction = float(np.clip(prediction, 1, 5))

        st.metric("Predicted burnout score", f"{prediction:.2f} / 5")
        if prediction >= 4:
            st.error("Model estimate is in the higher burnout range. This is a model output, not a diagnosis.")
        elif prediction >= 3:
            st.warning("Model estimate is in the moderate range.")
        else:
            st.success("Model estimate is in the lower range.")

        st.caption("The prediction is an ML estimate based on the selected model and the dataset; it is not a clinical assessment.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("MAI511-2 • Advanced Machine Learning • Regularized Regression Lab • Built for demonstration and viva")
