# ============================================================
# Streamlit GUI Dashboard
# Automated Classification of Consumer Financial Complaints using NLP
# ============================================================
# IMPORTANT FIX:
# This version does NOT crash if torch is missing.
# Dataset analysis and model-output dashboard work without torch.
# Live prediction is enabled only if torch + transformers + model weights exist.
# ============================================================

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Consumer Financial Complaints NLP Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
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

DEFAULT_MODEL_PATH = "finetuned_distilbert"
DEFAULT_METADATA_PATH = "assets/pipeline_metadata.json"
DEFAULT_CM_IMAGE_PATH = "assets/confusion_matrix.png"

BASE_TOKENIZER = "distilbert-base-uncased"
MAX_LEN = 512


# ============================================================
# 3. Optional Torch / Transformers Import
# ============================================================

TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
torch = None
DistilBertTokenizerFast = None
DistilBertForSequenceClassification = None

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


# ============================================================
# 4. Confusion Matrix Output from Fine-tuned DistilBERT
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
# 5. Text Cleaning
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


def show_missing_values(df):
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    missing_df["Missing %"] = (missing_df["Missing Values"] / len(df) * 100).round(2)
    missing_df = missing_df.sort_values("Missing Values", ascending=False)
    st.dataframe(missing_df, use_container_width=True)


def safe_value_counts(df, column, top_n=10):
    if column not in df.columns:
        return None

    counts = (
        df[column]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    counts.columns = [column, "Count"]

    return counts


def plot_top_categories(df, column, title, top_n=10):
    counts = safe_value_counts(df, column, top_n=top_n)

    if counts is None or counts.empty:
        st.info(f"Column not found or empty: {column}")
        return

    fig = px.bar(
        counts.sort_values("Count", ascending=True),
        x="Count",
        y=column,
        orientation="h",
        title=title
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 6. Metadata and Metrics
# ============================================================

def load_metadata(metadata_path):
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        return None, f"Metadata file not found: {metadata_path}"

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if "label_mapping" not in metadata:
            return None, "The metadata file does not contain 'label_mapping'."

        return metadata, None

    except Exception as e:
        return None, f"Could not load metadata file: {e}"


def get_label_maps(metadata):
    label_mapping = metadata["label_mapping"]

    label_to_id = {
        str(label): int(idx)
        for label, idx in label_mapping.items()
    }

    id_to_label = {
        int(idx): str(label)
        for label, idx in label_to_id.items()
    }

    return label_to_id, id_to_label


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
    accuracy = true_positive.sum() / total
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    summary = {
        "Total Test Records": int(total),
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1": weighted_f1
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


def plot_confusion_matrix_heatmap(cm, labels):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hovertemplate=(
                "True: %{y}<br>"
                "Predicted: %{x}<br>"
                "Count: %{z}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Confusion Matrix — Fine-tuned DistilBERT",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        height=800,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(autorange="reversed")

    return fig


# ============================================================
# 7. Optional Prediction Functions
# ============================================================

def can_run_live_prediction():
    return TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE


def token_aware_head_tail(text, tokenizer, max_len=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_len - 2:
        return tokenizer.encode(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True
        )

    head = tokens[:255]
    tail = tokens[-255:]

    input_ids = [tokenizer.cls_token_id] + head + tail + [tokenizer.sep_token_id]

    if len(input_ids) < max_len:
        input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))

    return input_ids[:max_len]


@st.cache_resource(show_spinner=False)
def load_tokenizer_and_model(model_path):
    if not can_run_live_prediction():
        return None, None, "torch or transformers is not installed."

    model_path = Path(model_path)

    if not model_path.exists():
        return None, None, f"Model folder not found: {model_path}"

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
        tokenizer.add_special_tokens({
            "additional_special_tokens": ["[REDACTED]"]
        })

        model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
        model.eval()

        return tokenizer, model, None

    except Exception as e:
        return None, None, f"Model loading failed: {e}"


def predict_single_text(text, model, tokenizer, id_to_label):
    cleaned_text = clean_text_adaptive(text)
    input_ids = token_aware_head_tail(cleaned_text, tokenizer, max_len=MAX_LEN)

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device)
        )

        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))

    predicted_label = id_to_label[pred_id]
    confidence = float(probs[pred_id])

    prob_df = pd.DataFrame({
        "Product": [id_to_label[i] for i in sorted(id_to_label.keys())],
        "Probability": probs
    })

    return predicted_label, confidence, cleaned_text, prob_df


# ============================================================
# 8. Sidebar
# ============================================================

st.sidebar.title("⚙️ Settings")

model_path = st.sidebar.text_input(
    "Fine-tuned model folder",
    value=DEFAULT_MODEL_PATH
)

metadata_path = st.sidebar.text_input(
    "Metadata file path",
    value=DEFAULT_METADATA_PATH
)

cm_image_path = st.sidebar.text_input(
    "Confusion matrix image path",
    value=DEFAULT_CM_IMAGE_PATH
)

st.sidebar.markdown("---")

if TORCH_AVAILABLE:
    st.sidebar.success("torch installed")
else:
    st.sidebar.warning("torch not installed")

if TRANSFORMERS_AVAILABLE:
    st.sidebar.success("transformers installed")
else:
    st.sidebar.warning("transformers not installed")


# ============================================================
# 9. Header
# ============================================================

st.title("💬 Automated Classification of Consumer Financial Complaints using NLP")

st.markdown(
    """
    This dashboard presents dataset analysis and Fine-tuned DistilBERT model-output
    analysis for consumer financial complaint classification.
    """
)


# ============================================================
# 10. Load Metadata
# ============================================================

metadata, metadata_error = load_metadata(metadata_path)

if metadata is not None:
    label_to_id, id_to_label = get_label_maps(metadata)
    labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
else:
    labels = DEFAULT_LABELS
    id_to_label = {i: label for i, label in enumerate(labels)}

tokenizer, model, prediction_load_error = load_tokenizer_and_model(model_path)

model_ready = tokenizer is not None and model is not None

if model_ready:
    st.success("Live prediction model loaded successfully.")
else:
    st.info(
        "Dataset analysis and model-output dashboard are available. "
        "Live prediction will be enabled only after torch, transformers, and the full model folder are available."
    )


# ============================================================
# 11. Tabs
# ============================================================

tab_data, tab_outputs, tab_processed, tab_single, tab_about = st.tabs([
    "📁 Original Dataset Analysis",
    "📈 NLP Model Output Dashboard",
    "🧪 Processed Splits",
    "🔍 Single Prediction",
    "ℹ️ About"
])


# ============================================================
# 12. Original Dataset Analysis
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
            narratives_available = (
                analysis_df[TEXT_COL].astype(str).str.strip().ne("").sum()
                if TEXT_COL in analysis_df.columns else 0
            )
            st.metric("Narratives Available", f"{narratives_available:,}")

        with col4:
            unique_products = (
                analysis_df[LABEL_COL].nunique()
                if LABEL_COL in analysis_df.columns else "N/A"
            )
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

            fig_len = px.histogram(
                analysis_df,
                x="Narrative_Word_Count",
                nbins=40,
                title="Distribution of Complaint Narrative Word Count"
            )
            st.plotly_chart(fig_len, use_container_width=True)

        st.markdown("### Business Category Analysis")

        col1, col2 = st.columns(2)

        with col1:
            plot_top_categories(analysis_df, LABEL_COL, "Top Product Categories", top_n=12)

        with col2:
            plot_top_categories(analysis_df, ISSUE_COL, "Top Complaint Issues", top_n=12)

        col1, col2 = st.columns(2)

        with col1:
            plot_top_categories(analysis_df, COMPANY_COL, "Top Companies by Complaint Count", top_n=12)

        with col2:
            plot_top_categories(analysis_df, STATE_COL, "Top States by Complaint Count", top_n=12)

        if DATE_COL in analysis_df.columns:
            date_df = analysis_df.dropna(subset=[DATE_COL]).copy()

            if not date_df.empty:
                date_df["Month"] = date_df[DATE_COL].dt.to_period("M").astype(str)

                monthly_counts = (
                    date_df
                    .groupby("Month")
                    .size()
                    .reset_index(name="Complaint Count")
                )

                st.markdown("### Complaint Trend Over Time")

                fig_time = px.line(
                    monthly_counts,
                    x="Month",
                    y="Complaint Count",
                    markers=True,
                    title="Monthly Complaint Volume"
                )

                st.plotly_chart(fig_time, use_container_width=True)

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
# 13. NLP Model Output Dashboard
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

    st.markdown("### Interactive Confusion Matrix")

    fig_cm = plot_confusion_matrix_heatmap(cm, labels)
    st.plotly_chart(fig_cm, use_container_width=True)

    if Path(cm_image_path).exists():
        with st.expander("View Original Confusion Matrix Image"):
            st.image(Image.open(cm_image_path), caption="Original confusion matrix output")

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

    fig_f1 = px.bar(
        metrics_df.sort_values("F1-score", ascending=True),
        x="F1-score",
        y="Product Category",
        orientation="h",
        title="F1-score by Product Category"
    )

    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("### Biggest Misclassification Patterns")

    st.dataframe(misclassification_df, use_container_width=True)

    fig_errors = px.bar(
        misclassification_df.sort_values("Count", ascending=True),
        x="Count",
        y="True Product",
        color="Predicted Product",
        orientation="h",
        title="Top Misclassification Pairs"
    )

    st.plotly_chart(fig_errors, use_container_width=True)

    if metadata is not None and "class_weights" in metadata:
        st.markdown("### Class Weights Used During Training")

        weights_df = (
            pd.DataFrame({
                "Product Category": list(metadata["class_weights"].keys()),
                "Class Weight": list(metadata["class_weights"].values())
            })
            .sort_values("Class Weight", ascending=True)
        )

        fig_weights = px.bar(
            weights_df,
            x="Class Weight",
            y="Product Category",
            orientation="h",
            title="Class Weight Multipliers"
        )

        st.plotly_chart(fig_weights, use_container_width=True)

        st.dataframe(weights_df, use_container_width=True)


# ============================================================
# 14. Processed Splits Dashboard
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

        fig_split = px.bar(
            split_summary,
            x="Split",
            y="Rows",
            title="Processed Dataset Split Sizes",
            text="Rows"
        )

        st.plotly_chart(fig_split, use_container_width=True)

        selected_split = st.selectbox(
            "Select split to preview",
            options=list(split_frames.keys())
        )

        selected_df = split_frames[selected_split]

        st.markdown(f"### {selected_split} Data Preview")
        st.dataframe(selected_df.head(20), use_container_width=True)

        if LABEL_COL in selected_df.columns:
            st.markdown(f"### {selected_split} Product Distribution")
            plot_top_categories(
                selected_df,
                LABEL_COL,
                f"{selected_split} Product Distribution",
                top_n=15
            )

        if "label" in selected_df.columns:
            st.markdown(f"### {selected_split} Encoded Label Distribution")
            plot_top_categories(
                selected_df,
                "label",
                f"{selected_split} Encoded Label Distribution",
                top_n=15
            )
    else:
        st.info("Upload any processed split file to analyse the train/validation/test outputs.")


# ============================================================
# 15. Single Prediction
# ============================================================

with tab_single:
    st.subheader("Single Complaint Classification")

    if not model_ready:
        st.warning("Live prediction is disabled.")
        if not TORCH_AVAILABLE:
            st.error("torch is not installed. Add torch to requirements.txt for live prediction.")
        if not TRANSFORMERS_AVAILABLE:
            st.error("transformers is not installed. Add transformers to requirements.txt for live prediction.")
        if prediction_load_error:
            st.info(prediction_load_error)

        st.markdown(
            """
            The rest of the dashboard still works because it uses saved model outputs.
            For live prediction, install torch and upload the full model weights folder.
            """
        )

    else:
        sample_text = (
            "I contacted the bank multiple times because there was an incorrect charge "
            "on my credit card account. The company did not investigate properly and "
            "kept reporting the balance as unpaid."
        )

        user_text = st.text_area(
            "Enter a consumer complaint narrative",
            value=sample_text,
            height=180
        )

        if st.button("Classify Complaint", type="primary"):
            if not user_text.strip():
                st.warning("Please enter a complaint narrative.")
            else:
                predicted_label, confidence, cleaned_text, prob_df = predict_single_text(
                    user_text,
                    model,
                    tokenizer,
                    id_to_label
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted Product", predicted_label)

                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")

                st.markdown("#### Cleaned Complaint Text")
                st.write(cleaned_text)

                st.markdown("#### Class Probability Distribution")

                fig = px.bar(
                    prob_df.sort_values("Probability", ascending=True),
                    x="Probability",
                    y="Product",
                    orientation="h",
                    title="Prediction Probabilities"
                )

                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 16. About
# ============================================================

with tab_about:
    st.subheader("About This Dashboard")

    st.markdown(
        """
        This version fixes the `ModuleNotFoundError: torch` issue by making torch optional.

        What works without torch:

        - Original dataset upload and analysis
        - Processed train/validation/test split analysis
        - Fine-tuned DistilBERT confusion matrix dashboard
        - Accuracy, macro F1, weighted F1
        - Per-class precision, recall, F1-score
        - Misclassification analysis
        - Class-weight visualisation

        What requires torch:

        - Live single complaint prediction
        - Batch prediction using the trained DistilBERT model
        """
    )

    st.markdown("#### For Streamlit Cloud")

    st.code(
        """
Make sure these files are in the same GitHub repository root:

streamlit_app.py
requirements.txt
assets/pipeline_metadata.json
assets/confusion_matrix.png

For live prediction, also add:

finetuned_distilbert/
    config.json
    model.safetensors or pytorch_model.bin
    tokenizer.json
    tokenizer_config.json
        """,
        language="text"
    )
