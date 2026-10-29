
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PATHS AND PAGE CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. DARK FUTURISTIC THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #090e12;
        --panel: #11191e;
        --border: rgba(159, 194, 190, 0.14);
        --text: #e5eeec;
        --muted: #849794;
        --accent: #83d8bb;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%,
                rgba(62, 123, 108, 0.16), transparent 35%),
            radial-gradient(ellipse at 5% 45%,
                rgba(43, 75, 87, 0.12), transparent 35%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .eyebrow {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #a8c8c0;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        margin-bottom: 20px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        background: var(--accent);
        box-shadow: 0 0 12px rgba(131, 216, 187, .65);
    }

    /* HERO PANEL */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 40px;
        border: 1px solid var(--border);
        border-radius: 22px;
        margin-bottom: 28px;
        background:
            radial-gradient(circle at 90% 15%,
                rgba(131, 216, 187, .10), transparent 36%),
            linear-gradient(130deg, #19272c, #0e151b);
        box-shadow:
            0 24px 65px rgba(0, 0, 0, .20),
            inset 0 1px 0 rgba(255, 255, 255, .025);
    }

    .hero-kicker {
        color: var(--accent);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        margin-bottom: 18px;
    }

    .hero-title {
        color: #edf4f1;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(32px, 4vw, 54px);
        font-weight: 800;
        line-height: 1.16;
        letter-spacing: -2px;
        margin: 0;
    }

    .hero-title span {
        color: var(--accent);
    }

    .hero-description {
        color: #9baea9;
        font-size: 14px;
        line-height: 1.9;
        max-width: 650px;
        margin-top: 18px;
    }

    .hero-footer {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 32px;
        padding-top: 22px;
        border-top: 1px solid var(--border);
    }

    .hero-chip {
        color: #a5bbb5;
        background: rgba(255, 255, 255, .025);
        border: 1px solid rgba(159, 194, 190, .15);
        padding: 9px 12px;
        border-radius: 7px;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
    }

    /* SECTION HEADINGS */

    .section-heading {
        color: #e4eeeb;
        font-family: 'Manrope', sans-serif;
        font-size: 22px;
        font-weight: 700;
        margin: 8px 0 5px;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 22px;
        line-height: 1.8;
    }

    /* METRIC CARDS */

    .kpi-card {
        background: linear-gradient(145deg, #19272b, #10181d);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 22px;
        min-height: 125px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, .12);
    }

    .kpi-label {
        color: #91a59f;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .kpi-value {
        color: #edf4f1;
        font-family: 'Manrope', sans-serif;
        font-size: 29px;
        font-weight: 800;
        margin-top: 12px;
    }

    .kpi-note {
        color: #849993;
        font-size: 11px;
        margin-top: 5px;
    }

    .panel {
        background: rgba(17, 26, 31, .82);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .panel-title {
        color: #dfeae6;
        font-family: 'Manrope', sans-serif;
        font-size: 15px;
        font-weight: 700;
    }

    .panel-description {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.8;
        margin-top: 5px;
    }

    /* TABS */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(14, 21, 25, .78);
        padding: 7px;
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 15px;
        color: #91a59f;
        border-radius: 8px;
        font-size: 12px;
    }

    .stTabs [aria-selected="true"] {
        color: #e5f5ee !important;
        background: rgba(131, 216, 187, .10) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: var(--accent) !important;
    }

    /* INPUTS AND BUTTONS */

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #121d22 !important;
        color: #e5eeec !important;
        border-color: var(--border) !important;
        border-radius: 9px !important;
    }

    .stFileUploader section {
        background: rgba(17, 26, 31, .65);
        border: 1px dashed rgba(131, 216, 187, .3);
        border-radius: 13px;
    }

    .stButton button,
    .stDownloadButton button {
        background: linear-gradient(135deg, #a0e2ca, #6fbda2);
        color: #10251e;
        border: 0;
        border-radius: 9px;
        font-weight: 700;
        padding: 10px 16px;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        background: #b7efd9;
        color: #10251e;
        border: 0;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border);
    }

    .app-footer {
        text-align: center;
        color: #61746f;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1px;
        padding-top: 25px;
        margin-top: 35px;
        border-top: 1px solid var(--border);
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 25px 20px;
        }

        .hero-title {
            letter-spacing: -1px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. HERO HEADER — HTML IS RENDERED HERE
# =========================================================


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            RETENTION ANALYTICS PLATFORM &nbsp; / &nbsp; 01
        </div>
        <div class="hero-title">
            Understand churn.<br>
            <span>Protect customer value.</span>
        </div>
        <div class="hero-description">
            Explore customer behavior, evaluate machine-learning
            performance, and identify customers who may need
            retention support.
        </div>
        <div class="hero-footer">
            <span class="hero-chip">01 &nbsp; OVERVIEW</span>
            <span class="hero-chip">02 &nbsp; ANALYTICS</span>
            <span class="hero-chip">03 &nbsp; MODEL INSIGHTS</span>
            <span class="hero-chip">04 &nbsp; PREDICTIONS</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 4. LOAD PROJECT FILES
# =========================================================

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_comparison():
    if not COMPARISON_PATH.exists():
        return None
    return pd.read_csv(COMPARISON_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def churn_labels(series):
    values = series.astype("string").str.strip().str.lower()
    return values.map({
        "yes": "Yes",
        "no": "No",
        "1": "Yes",
        "0": "No",
        "true": "Yes",
        "false": "No",
    })


def get_model_features(loaded_model):
    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)

    steps = getattr(loaded_model, "named_steps", {})
    preprocessor = steps.get("preprocessor")

    if preprocessor is not None and hasattr(
        preprocessor, "feature_names_in_"
    ):
        return list(preprocessor.feature_names_in_)

    raise ValueError(
        "Could not identify the trained model's input columns."
    )


def kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def heading(title, subtitle):
    st.markdown(
        f"""
        <div class="section-heading">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


df = load_dataset()
comparison = load_comparison()
model = load_model()


# =========================================================
# 5. VALIDATE THE DATASET
# =========================================================

if df is None:
    st.error(
        "Dataset not found. Put Telco-Customer-Churn.csv "
        "inside the data folder."
    )
    st.stop()

if "Churn" not in df.columns:
    st.error("The dataset must contain a Churn column.")
    st.stop()

labels = churn_labels(df["Churn"])
valid_labels = labels.dropna()

total_customers = len(df)
churned = int(valid_labels.eq("Yes").sum())
retained = int(valid_labels.eq("No").sum())
valid_total = len(valid_labels)

churn_rate = churned / valid_total * 100 if valid_total else 0
retention_rate = retained / valid_total * 100 if valid_total else 0


# =========================================================
# 6. APPLICATION TABS
# =========================================================

overview, analytics, evaluation, predictions = st.tabs([
    "◈  Overview",
    "⌁  Churn Analytics",
    "◎  Model Evaluation",
    "↗  Predict Churn",
])


# =========================================================
# 7. OVERVIEW
# =========================================================

with overview:
    heading(
        "Business overview",
        "A high-level view of customer retention and churn behavior.",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi("Total customers", f"{total_customers:,}",
            "Records in dataset")

    with c2:
        kpi("Churned customers", f"{churned:,}",
            "Customers who left")

    with c3:
        kpi("Churn rate", f"{churn_rate:.1f}%",
            "Of customers with known labels")

    with c4:
        kpi("Retention rate", f"{retention_rate:.1f}%",
            "Of customers with known labels")

    st.markdown("")

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Customer distribution</div>
                <div class="panel-description">
                    Retained customers compared with churned customers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        distribution = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [retained, churned],
        })

        st.bar_chart(distribution.set_index("Status"))

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Dataset information</div>
                <div class="panel-description">
                    Basic information about the loaded dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        a, b = st.columns(2)

        with a:
            st.metric("Rows", f"{df.shape[0]:,}")

        with b:
            st.metric("Columns", f"{df.shape[1]:,}")

        st.metric(
            "Missing values",
            f"{int(df.isna().sum().sum()):,}",
        )

        if "tenure" in df.columns:
            tenure = pd.to_numeric(df["tenure"], errors="coerce")
            st.metric(
                "Average tenure",
                f"{tenure.mean():.1f} months"
                if tenure.notna().any()
                else "N/A",
            )

    st.markdown("---")
    heading(
        "Dataset preview",
        "Preview the first ten customer records.",
    )

    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# =========================================================
# 8. CHURN ANALYTICS
# =========================================================

with analytics:
    heading(
        "Churn analytics",
        "Explore customer segments and discover churn patterns.",
    )

    analysis_df = df.copy()
    analysis_df["_ChurnStatus"] = labels

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Churn distribution</div>
                <div class="panel-description">
                    Number of retained and churned customers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        counts = labels.value_counts().reindex(
            ["No", "Yes"], fill_value=0
        )

        st.bar_chart(pd.DataFrame({
            "Customers": [
                int(counts.get("No", 0)),
                int(counts.get("Yes", 0)),
            ]
        }, index=["Retained", "Churned"]))

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Churn by contract</div>
                <div class="panel-description">
                    Compare churn rates for different contract types.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "Contract" in analysis_df.columns:
            contract_data = analysis_df.dropna(
                subset=["Contract", "_ChurnStatus"]
            )

            if not contract_data.empty:
                contract_rates = (
                    contract_data.groupby("Contract")["_ChurnStatus"]
                    .apply(lambda x: x.eq("Yes").mean() * 100)
                    .sort_values(ascending=False)
                    .rename("Churn rate (%)")
                )
                st.bar_chart(contract_rates)
            else:
                st.info("No usable contract data is available.")
        else:
            st.info("The dataset has no Contract column.")

    st.markdown("---")

    heading(
        "Customer segment explorer",
        "Choose an attribute to compare segment-level churn rates.",
    )

    candidates = [
        "Contract", "InternetService", "PaymentMethod",
        "TechSupport", "OnlineSecurity", "PaperlessBilling",
        "SeniorCitizen", "Partner", "Dependents", "gender",
    ]

    available = [c for c in candidates if c in df.columns]

    if available:
        segment = st.selectbox("Customer attribute", available)

        segment_data = analysis_df.dropna(
            subset=[segment, "_ChurnStatus"]
        )

        if not segment_data.empty:
            summary = (
                segment_data.groupby(segment)["_ChurnStatus"]
                .agg(
                    Customers="count",
                    Churned=lambda x: x.eq("Yes").sum(),
                )
            )

            summary["Churn rate (%)"] = (
                summary["Churned"] / summary["Customers"] * 100
            ).round(2)

            summary = summary.sort_values(
                "Churn rate (%)", ascending=False
            )

            st.bar_chart(summary[["Churn rate (%)"]])
            st.dataframe(
                summary.reset_index(),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No usable data for the selected segment.")
    else:
        st.info("No supported segment columns were found.")


# =========================================================
# 9. MODEL EVALUATION
# =========================================================

with evaluation:
    heading(
        "Model evaluation",
        "Compare the performance metrics recorded during model evaluation.",
    )

    if comparison is None:
        st.warning(
            "model_comparison.csv was not found in the models folder."
        )

    else:
        required = [
            "Model", "Accuracy", "Precision",
            "Recall", "F1 Score", "ROC-AUC",
        ]

        missing = [c for c in required if c not in comparison.columns]

        if missing:
            st.error(
                "Missing columns in model_comparison.csv: "
                + ", ".join(missing)
            )

        else:
            comparison_data = comparison.copy()

            metrics = [
                "Accuracy", "Precision", "Recall",
                "F1 Score", "ROC-AUC",
            ]

            for metric in metrics:
                comparison_data[metric] = pd.to_numeric(
                    comparison_data[metric], errors="coerce"
                )

            display_data = comparison_data.copy()

            for metric in metrics:
                display_data[metric] = display_data[metric].map(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
                )

            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">Model performance comparison</div>
                    <div class="panel-description">
                        Results loaded from your saved comparison CSV.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
            )

            left, right = st.columns(2, gap="large")

            with left:
                heading("Accuracy", "Overall proportion of correct predictions.")

                st.bar_chart(
                    comparison_data[["Model", "Accuracy"]]
                    .dropna()
                    .set_index("Model")
                )

            with right:
                heading("Recall", "Ability to identify actual churn cases.")

                st.bar_chart(
                    comparison_data[["Model", "Recall"]]
                    .dropna()
                    .set_index("Model")
                )

            st.markdown("---")

            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">Metric guide</div>
                    <div class="panel-description">
                        <b>Accuracy:</b> Overall correctness.<br><br>
                        <b>Precision:</b> How often predicted churn cases
                        are actually churn cases.<br><br>
                        <b>Recall:</b> How many actual churn cases
                        the model identifies.<br><br>
                        <b>F1 score:</b> Balance between precision and recall.<br><br>
                        <b>ROC-AUC:</b> How well the model separates
                        churned and retained customers across thresholds.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# 10. PREDICT CHURN
# =========================================================

with predictions:
    heading(
        "Churn prediction",
        "Upload customer records and estimate their churn probabilities.",
    )

    if model is None:
        st.error(
            "Trained model not found. Put churn_model.joblib "
            "inside the models folder."
        )

    else:
        try:
            features = get_model_features(model)
        except Exception as exc:
            st.error(f"Could not inspect model features: {exc}")
            st.stop()

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Batch prediction workspace</div>
                <div class="panel-description">
                    Upload a CSV containing all the trained model's input
                    features. The application will score each record and
                    assign a risk band based on its churn probability.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View required input columns"):
            st.write("Your uploaded CSV must include:")
            st.code(", ".join(features))

        uploaded = st.file_uploader(
            "Upload customer CSV",
            type=["csv"],
            key="churn_upload",
        )

        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Unable to read CSV: {exc}")
                st.stop()

            st.caption(
                f"{len(input_df):,} rows · {len(input_df.columns):,} columns"
            )

            st.dataframe(
                input_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            missing_features = [
                feature for feature in features
                if feature not in input_df.columns
            ]

            if missing_features:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_features)
                )

            elif input_df.empty:
                st.warning("The uploaded file contains no records.")

            elif st.button(
                "Run churn prediction",
                type="primary",
                use_container_width=True,
            ):
                try:
                    model_input = input_df[features].copy()

                    for column in model_input.select_dtypes(
                        include=["object", "string"]
                    ).columns:
                        model_input[column] = (
                            model_input[column]
                            .astype("string")
                            .str.strip()
                            .replace("", pd.NA)
                        )

                    if "TotalCharges" in model_input.columns:
                        model_input["TotalCharges"] = pd.to_numeric(
                            model_input["TotalCharges"],
                            errors="coerce",
                        )

                    if not hasattr(model, "predict_proba"):
                        raise ValueError(
                            "The trained model does not support "
                            "predict_proba."
                        )

                    predictions_array = model.predict(model_input)
                    probabilities = model.predict_proba(model_input)

                    classes = list(getattr(model, "classes_", []))

                    if not classes and hasattr(model, "named_steps"):
                        final_estimator = list(
                            model.named_steps.values()
                        )[-1]
                        classes = list(
                            getattr(final_estimator, "classes_", [])
                        )

                    churn_index = None

                    for index, class_value in enumerate(classes):
                        if str(class_value).strip().lower() in {
                            "1", "yes", "true"
                        }:
                            churn_index = index
                            break

                    if churn_index is None:
                        raise ValueError(
                            "Could not identify the positive churn class. "
                            "Expected class label 1 or Yes."
                        )

                    churn_prob = probabilities[:, churn_index]

                    output = input_df.copy()

                    output["Predicted Churn"] = [
                        "Yes"
                        if str(value).strip().lower()
                        in {"1", "yes", "true"}
                        else "No"
                        for value in predictions_array
                    ]

                    output["Churn Probability (%)"] = (
                        churn_prob * 100
                    ).round(2)

                    output["Risk Level"] = pd.cut(
                        churn_prob,
                        bins=[-0.001, 0.30, 0.60, 1.001],
                        labels=["Low", "Medium", "High"],
                        include_lowest=True,
                    ).astype("string")

                    st.session_state["churn_results"] = output

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.info(
                        "Verify that the uploaded data has the correct "
                        "column names, data types and category values."
                    )

        results = st.session_state.get("churn_results")

        if results is not None:
            st.markdown("---")

            heading(
                "Prediction results",
                "Review and filter the model's estimated churn risk.",
            )

            total_scored = len(results)
            predicted_count = int(
                results["Predicted Churn"].eq("Yes").sum()
            )
            high_risk = int(results["Risk Level"].eq("High").sum())
            mean_probability = (
                results["Churn Probability (%)"].mean()
                if total_scored else 0
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                kpi("Records scored", f"{total_scored:,}", "Processed records")

            with c2:
                kpi("Predicted churn", f"{predicted_count:,}", "Classified as churn")

            with c3:
                kpi("High risk", f"{high_risk:,}", "Probability above 60%")

            with c4:
                kpi("Average probability", f"{mean_probability:.1f}%",
                    "Across uploaded records")

            filter_col, sort_col = st.columns(2)

            with filter_col:
                risk_filter = st.selectbox(
                    "Filter risk",
                    ["All", "High", "Medium", "Low"],
                )

            with sort_col:
                sort_order = st.selectbox(
                    "Sort by probability",
                    ["Highest first", "Lowest first"],
                )

            shown = results.copy()

            if risk_filter != "All":
                shown = shown[shown["Risk Level"] == risk_filter]

            shown = shown.sort_values(
                "Churn Probability (%)",
                ascending=(sort_order == "Lowest first"),
            )

            st.dataframe(
                shown,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download all predictions as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.caption(
                "Risk bands: Low ≤ 30%, Medium > 30% to 60%, "
                "High > 60%. These are configurable thresholds, "
                "not guarantees of customer behavior."
            )


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown(
    """
    <div class="app-footer">
        CUSTOMER INTELLIGENCE &nbsp; / &nbsp;
        CHURN PREDICTION SYSTEM &nbsp; / &nbsp;
        MACHINE LEARNING ANALYTICS
    </div>
    """,
    unsafe_allow_html=True,
)
