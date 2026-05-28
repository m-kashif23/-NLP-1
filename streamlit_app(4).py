# ============================================================
# Streamlit Dashboard
# Automated Classification of Consumer Financial Complaints using NLP
# ============================================================
# Safe Streamlit Cloud Version:
# - No torch import at startup
# - No plotly dependency
# - Uses Streamlit native charts
# - Works for dataset analysis and saved model-output dashboard
# ============================================================

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Consumer Financial Complaints NLP Dashboard",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# 2. Constants
# ============================================================

TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"
ID_COL = "Complaint ID"

DATE_COL = "Date received"
ISSUE_COL = "Issue"
COMPANY_COL = "Company"
STATE_COL = "State"
SUBMITTED_COL = "Submitted via"
TIMELY_COL = "Timely response?"

DEFAULT_METADATA_PATH = "assets/pipeline_metadata.json"
DEFAULT_CM_IMAGE_PATH = "assets/confusion_matrix.png"


# ============================================================
# 3. Saved Model Output
# ============================================================

DEFAULT_LABELS = [
    "Checking or savings account",
    "Credit card",
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Debt or credit management",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Student loan",
    "Vehicle loan or lease"
]

DEFAULT_CONFUSION_MATRIX = np.array([
    [1211,  81,   17,   11,  6, 170,  4,  16, 14,   0,   3],
    [  80,1173,   68,   59,  6,  24,  7,  23, 10,   3,   5],
    [  42, 160, 8474,  537, 12,   8, 34,  34,  0,  64, 132],
    [  20,  37,  209, 2104,  9,   5,  7,  32,  3,   6,  42],
    [   3,   8,    5,   28, 24,   1,  2,   4,  0,   1,   4],
    [ 130,  17,    1,    5,  1, 639,  2,  14,  9,   0,   2],
    [   3,   3,    8,   10,  1,   1,464,  15,  0,   8,   2],
    [   6,  16,   16,   18,  3,   8,  6, 213,  1,   5,  14],
    [  27,   4,    0,    1,  0,  20,  0,   0, 74,   1,   0],
    [   1,   0,    8,    6,  0,   0,  3,   7,  0, 227,   1],
    [   3,   2,   23,   10,  0,   0,  3,  17,  0,   2, 387]
])


# ============================================================
# 4. Helper Functions
# ============================================================

def clean_text_adaptive(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"X{2,}", "[REDACTED]", text)
    text = re.sub(r"\d{2}/\d{2}/\d{4}", "[REDACTED]", text)
    text = re.sub(r"X{1,2}/X{1,2}/X{2,4}", "[REDACTED]", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def add_cleaning_columns(df):
    work_df = df.copy()

    if TEXT_COL in work_df.columns:
        work_df[TEXT_COL] = work_df[TEXT_COL].fillna("").astype(str)
        work_df["Cleaned_Text"] = work_df[TEXT_COL].apply(clean_text_adaptive)
        work_df["Narrative_Word_Count"] = work_df["Cleaned_Text"].apply(lambda x: len(x.split()))
        work_df["Narrative_Character_Count"] = work_df["Cleaned_Text"].apply(len)

    if DATE_COL in work_df.columns:
        work_df[DATE_COL] = pd.to_datetime(work_df[DATE_COL], errors="coerce")

    return work_df


def load_metadata(metadata_path):
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_labels_from_metadata(metadata):
    if metadata is None or "label_mapping" not in metadata:
        return DEFAULT_LABELS

    mapping = metadata["label_mapping"]
    id_to_label = {int(v): k for k, v in mapping.items()}
    return [id_to_label[i] for i in sorted(id_to_label.keys())]


def value_counts_df(df, column, top_n=15):
    if column not in df.columns:
        return None

    out = (
        df[column]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    out.columns = [column, "Count"]
    return out


def show_bar_chart_from_counts(df, label_column):
    if df is None or df.empty:
        st.info("No data available for this chart.")
        return

    chart_df = df.set_index(label_column)
    st.bar_chart(chart_df)


def show_missing_values(df):
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    missing_df["Missing %"] = (missing_df["Missing Values"] / len(df) * 100).round(2)
    missing_df = missing_df.sort_values("Missing Values", ascending=False)
    st.dataframe(missing_df, use_container_width=True)


def compute_metrics_from_confusion_matrix(cm, labels):
    cm = np.array(cm)

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    true_positive = np.diag(cm)

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted != 0
    )

    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=float),
        where=support != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0
    )

    metrics_df = pd.DataFrame({
        "Product Category": labels,
        "Support": support,
        "Correct Predictions": true_positive,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    total = cm.sum()

    summary = {
        "Total Test Records": int(total),
        "Accuracy": true_positive.sum() / total,
        "Macro Precision": precision.mean(),
        "Macro Recall": recall.mean(),
        "Macro F1": f1.mean(),
        "Weighted Precision": np.average(precision, weights=support),
        "Weighted Recall": np.average(recall, weights=support),
        "Weighted F1": np.average(f1, weights=support)
    }

    return metrics_df, summary


def build_misclassification_table(cm, labels, top_n=15):
    rows = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "True Product": true_label,
                    "Predicted Product": pred_label,
                    "Count": int(cm[i, j])
                })

    return (
        pd.DataFrame(rows)
        .sort_values("Count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 5. Sidebar
# ============================================================

st.sidebar.title("⚙️ Settings")

metadata_path = st.sidebar.text_input(
    "Metadata file path",
    value=DEFAULT_METADATA_PATH
)

cm_image_path = st.sidebar.text_input(
    "Confusion matrix image path",
    value=DEFAULT_CM_IMAGE_PATH
)

st.sidebar.info(
    "This version avoids torch and plotly, so it is safer for Streamlit Cloud."
)


# ============================================================
# 6. Header
# ============================================================

st.title("💬 Automated Classification of Consumer Financial Complaints using NLP")

st.markdown(
    """
    This dashboard presents the original dataset analysis and saved Fine-tuned
    DistilBERT model-output analysis for consumer financial complaint classification.
    """
)


# ============================================================
# 7. Metadata
# ============================================================

metadata = load_metadata(metadata_path)
labels = get_labels_from_metadata(metadata)


# ============================================================
# 8. Tabs
# ============================================================

tab_data, tab_outputs, tab_processed, tab_about = st.tabs([
    "📁 Original Dataset Analysis",
    "📈 NLP Model Output Dashboard",
    "🧪 Processed Splits",
    "ℹ️ About"
])


# ============================================================
# 9. Original Dataset Analysis
# ============================================================

with tab_data:
    st.subheader("Upload Original Complaints Dataset")

    uploaded_dataset = st.file_uploader(
        "Upload the original complaints CSV file",
        type=["csv"],
        key="original_dataset_uploader"
    )

    if uploaded_dataset is not None:
        try:
            raw_df = pd.read_csv(uploaded_dataset)
            analysis_df = add_cleaning_columns(raw_df)
            st.session_state["analysis_df"] = analysis_df
            st.success("Dataset uploaded and prepared successfully.")
        except Exception as e:
            st.error(f"Could not read uploaded CSV file: {e}")

    if "analysis_df" in st.session_state:
        analysis_df = st.session_state["analysis_df"]

        st.markdown("### Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Rows", f"{analysis_df.shape[0]:,}")

        with col2:
            st.metric("Total Columns", analysis_df.shape[1])

        with col3:
            if TEXT_COL in analysis_df.columns:
                narratives_available = analysis_df[TEXT_COL].astype(str).str.strip().ne("").sum()
            else:
                narratives_available = 0
            st.metric("Narratives Available", f"{narratives_available:,}")

        with col4:
            if LABEL_COL in analysis_df.columns:
                unique_products = analysis_df[LABEL_COL].nunique()
            else:
                unique_products = "N/A"
            st.metric("Unique Products", unique_products)

        st.markdown("### Data Preview")
        st.dataframe(analysis_df.head(20), use_container_width=True)

        st.markdown("### Missing Values")
        show_missing_values(analysis_df)

        if "Narrative_Word_Count" in analysis_df.columns:
            st.markdown("### Complaint Narrative Length")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Average Word Count",
                    f"{analysis_df['Narrative_Word_Count'].mean():.1f}"
                )

            with col2:
                st.metric(
                    "Median Word Count",
                    f"{analysis_df['Narrative_Word_Count'].median():.1f}"
                )

            hist_counts = pd.cut(
                analysis_df["Narrative_Word_Count"],
                bins=20
            ).value_counts().sort_index()

            hist_df = pd.DataFrame({
                "Word Count Range": hist_counts.index.astype(str),
                "Count": hist_counts.values
            }).set_index("Word Count Range")

            st.bar_chart(hist_df)

        st.markdown("### Product Category Distribution")
        product_counts = value_counts_df(analysis_df, LABEL_COL, top_n=15)
        show_bar_chart_from_counts(product_counts, LABEL_COL)
        if product_counts is not None:
            st.dataframe(product_counts, use_container_width=True)

        st.markdown("### Top Complaint Issues")
        issue_counts = value_counts_df(analysis_df, ISSUE_COL, top_n=15)
        show_bar_chart_from_counts(issue_counts, ISSUE_COL)
        if issue_counts is not None:
            st.dataframe(issue_counts, use_container_width=True)

        st.markdown("### Top Companies")
        company_counts = value_counts_df(analysis_df, COMPANY_COL, top_n=15)
        show_bar_chart_from_counts(company_counts, COMPANY_COL)
        if company_counts is not None:
            st.dataframe(company_counts, use_container_width=True)

        st.markdown("### Top States")
        state_counts = value_counts_df(analysis_df, STATE_COL, top_n=15)
        show_bar_chart_from_counts(state_counts, STATE_COL)
        if state_counts is not None:
            st.dataframe(state_counts, use_container_width=True)

        if DATE_COL in analysis_df.columns:
            date_df = analysis_df.dropna(subset=[DATE_COL]).copy()

            if not date_df.empty:
                date_df["Month"] = date_df[DATE_COL].dt.to_period("M").astype(str)

                monthly_counts = (
                    date_df
                    .groupby("Month")
                    .size()
                    .reset_index(name="Complaint Count")
                    .set_index("Month")
                )

                st.markdown("### Monthly Complaint Trend")
                st.line_chart(monthly_counts)

        prepared_csv = analysis_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prepared Dataset",
            data=prepared_csv,
            file_name="prepared_consumer_complaints_dataset.csv",
            mime="text/csv"
        )

    else:
        st.info(f"Upload the original dataset. The key text column should be `{TEXT_COL}`.")


# ============================================================
# 10. NLP Model Output Dashboard
# ============================================================

with tab_outputs:
    st.subheader("Fine-tuned DistilBERT Output Dashboard")

    cm = DEFAULT_CONFUSION_MATRIX

    metrics_df, summary = compute_metrics_from_confusion_matrix(cm, labels)
    misclassification_df = build_misclassification_table(cm, labels, top_n=15)

    st.markdown("### Model Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Test Records", f"{summary['Total Test Records']:,}")

    with col2:
        st.metric("Accuracy", f"{summary['Accuracy']:.2%}")

    with col3:
        st.metric("Macro F1", f"{summary['Macro F1']:.2%}")

    with col4:
        st.metric("Weighted F1", f"{summary['Weighted F1']:.2%}")

    st.markdown("### Original Confusion Matrix Output")

    if Path(cm_image_path).exists():
        st.image(cm_image_path, caption="Confusion Matrix — Fine-tuned DistilBERT")
    else:
        st.info("Confusion matrix image not found. Showing numeric table instead.")

    st.markdown("### Confusion Matrix Table")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("### Per-Class Performance")
    st.dataframe(
        metrics_df.style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-score": "{:.3f}"
        }),
        use_container_width=True
    )

    st.markdown("### Per-Class F1-score")
    f1_chart = metrics_df[["Product Category", "F1-score"]].set_index("Product Category")
    st.bar_chart(f1_chart)

    st.markdown("### Biggest Misclassification Patterns")
    st.dataframe(misclassification_df, use_container_width=True)

    error_chart = (
        misclassification_df
        .assign(Pair=lambda x: x["True Product"] + " → " + x["Predicted Product"])
        [["Pair", "Count"]]
        .set_index("Pair")
    )
    st.bar_chart(error_chart)

    if metadata is not None and "class_weights" in metadata:
        st.markdown("### Class Weights Used During Training")

        weights_df = (
            pd.DataFrame({
                "Product Category": list(metadata["class_weights"].keys()),
                "Class Weight": list(metadata["class_weights"].values())
            })
            .sort_values("Class Weight", ascending=False)
        )

        st.dataframe(weights_df, use_container_width=True)
        st.bar_chart(weights_df.set_index("Product Category"))


# ============================================================
# 11. Processed Splits
# ============================================================

with tab_processed:
    st.subheader("Upload Processed Train / Validation / Test Files")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_file = st.file_uploader("Upload processed_train.csv", type=["csv"], key="train_split")

    with col2:
        val_file = st.file_uploader("Upload processed_val.csv", type=["csv"], key="val_split")

    with col3:
        test_file = st.file_uploader("Upload processed_test.csv", type=["csv"], key="test_split")

    split_frames = {}

    if train_file is not None:
        split_frames["Train"] = pd.read_csv(train_file)

    if val_file is not None:
        split_frames["Validation"] = pd.read_csv(val_file)

    if test_file is not None:
        split_frames["Test"] = pd.read_csv(test_file)

    if split_frames:
        split_summary = pd.DataFrame({
            "Split": list(split_frames.keys()),
            "Rows": [df.shape[0] for df in split_frames.values()],
            "Columns": [df.shape[1] for df in split_frames.values()]
        })

        st.markdown("### Split Summary")
        st.dataframe(split_summary, use_container_width=True)

        st.bar_chart(split_summary.set_index("Split")[["Rows"]])

        selected_split = st.selectbox(
            "Select split to preview",
            options=list(split_frames.keys())
        )

        selected_df = split_frames[selected_split]

        st.markdown(f"### {selected_split} Data Preview")
        st.dataframe(selected_df.head(20), use_container_width=True)

        if LABEL_COL in selected_df.columns:
            st.markdown(f"### {selected_split} Product Distribution")
            split_product_counts = value_counts_df(selected_df, LABEL_COL, top_n=15)
            show_bar_chart_from_counts(split_product_counts, LABEL_COL)
            if split_product_counts is not None:
                st.dataframe(split_product_counts, use_container_width=True)

        if "label" in selected_df.columns:
            st.markdown(f"### {selected_split} Encoded Label Distribution")
            split_label_counts = value_counts_df(selected_df, "label", top_n=15)
            show_bar_chart_from_counts(split_label_counts, "label")
            if split_label_counts is not None:
                st.dataframe(split_label_counts, use_container_width=True)
    else:
        st.info("Upload any processed split file to analyse the train/validation/test outputs.")


# ============================================================
# 12. About
# ============================================================

with tab_about:
    st.subheader("About This Version")

    st.markdown(
        """
        This is the safest Streamlit Cloud version.

        It removes both:

        - `torch`
        - `plotly`

        Therefore, it avoids the two errors you faced:

        - `ModuleNotFoundError: torch`
        - `ModuleNotFoundError: plotly`

        This version focuses on:

        - Original dataset analysis
        - Saved NLP model output dashboard
        - Confusion matrix analysis
        - Per-class model performance
        - Processed split analysis

        Live prediction is not included in this lightweight version because that
        requires `torch`, `transformers`, and the full fine-tuned model weights.
        """
    )

    st.markdown("#### Required files in GitHub repository")

    st.code(
        """
streamlit_app.py
requirements.txt
assets/pipeline_metadata.json
assets/confusion_matrix.png
        """,
        language="text"
    )
