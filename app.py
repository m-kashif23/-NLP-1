"""
Automated Classification of Consumer Financial Complaints using NLP
-------------------------------------------------------------------
Streamlit dashboard for a fine-tuned DistilBERT model that classifies
CFPB consumer-complaint narratives into 11 financial-product categories.

The trained model itself lives on the Hugging Face Hub
(default: Mkashif23/cfpb-distilbert); the dashboard downloads it on demand.

Supporting files expected next to app.py (commit them to the repo):
    config.json, tokenizer.json, tokenizer_config.json, pipeline_metadata.json
    training_curves.png, confusion_matrix.png
    processed_train.csv / processed_val.csv / processed_test.csv
    a raw complaints CSV  ->  optional, enables raw-text examples

Any file that isn't present simply hides that section of the page.

Run with:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Complaint Classifier Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px;}
        h1, h2, h3 {letter-spacing: -0.5px;}
        [data-testid="stMetric"] {
            background: rgba(130,150,200,0.08);
            border: 1px solid rgba(130,150,200,0.20);
            border-radius: 14px;
            padding: 16px 18px;
        }
        [data-testid="stMetricLabel"] {opacity: 0.75;}
        .pill {
            display:inline-block; padding:3px 12px; margin:2px 4px 2px 0;
            border-radius:999px; font-size:0.80rem;
            background:rgba(80,130,255,0.12); border:1px solid rgba(80,130,255,0.30);
        }
        .muted {opacity:0.7; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Static project knowledge (from the uploaded artifacts)
# --------------------------------------------------------------------------- #
LABEL_MAPPING = {
    "Checking or savings account": 0,
    "Credit card": 1,
    "Credit reporting or other personal consumer reports": 2,
    "Debt collection": 3,
    "Debt or credit management": 4,
    "Money transfer, virtual currency, or money service": 5,
    "Mortgage": 6,
    "Payday loan, title loan, personal loan, or advance loan": 7,
    "Prepaid card": 8,
    "Student loan": 9,
    "Vehicle loan or lease": 10,
}
ID2LABEL = {v: k for k, v in LABEL_MAPPING.items()}
LABELS_ORDERED = [ID2LABEL[i] for i in range(len(ID2LABEL))]

SHORT_LABELS = [
    "Checking/Savings", "Credit card", "Credit reporting", "Debt collection",
    "Debt/credit mgmt", "Money transfer", "Mortgage", "Payday/Personal loan",
    "Prepaid card", "Student loan", "Vehicle loan",
]

CONFUSION_MATRIX = np.array(
    [
        [1211,   81,   17,   11,    6,  170,    4,   16,   14,    0,    3],
        [  80, 1173,   68,   59,    6,   24,    7,   23,   10,    3,    5],
        [  42,  160, 8474,  537,   12,    8,   34,   34,    0,   64,  132],
        [  20,   37,  209, 2104,    9,    5,    7,   32,    3,    6,   42],
        [   3,    8,    5,   28,   24,    1,    2,    4,    0,    1,    4],
        [ 130,   17,    1,    5,    1,  639,    2,   14,    9,    0,    2],
        [   3,    3,    8,   10,    1,    1,  464,   15,    0,    8,    2],
        [   6,   16,   16,   18,    3,    8,    6,  213,    1,    5,   14],
        [  27,    4,    0,    1,    0,   20,    0,    0,   74,    1,    0],
        [   1,    0,    8,    6,    0,    0,    3,    7,    0,  227,    1],
        [   3,    2,   23,   10,    0,    0,    3,   17,    0,    2,  387],
    ],
    dtype=float,
)

CLASS_WEIGHTS = {
    "Checking or savings account": 1.0386,
    "Credit card": 1.0921,
    "Credit reporting or other personal consumer reports": 0.1676,
    "Debt collection": 0.6435,
    "Debt or credit management": 19.7611,
    "Money transfer, virtual currency, or money service": 1.9428,
    "Mortgage": 3.0875,
    "Payday loan, title loan, personal loan, or advance loan": 5.2018,
    "Prepaid card": 12.5583,
    "Student loan": 6.2828,
    "Vehicle loan or lease": 3.5562,
}

REDACTED_TOKEN_ID = 30522  # "[REDACTED]" extra special token used to mask PII

# Canonical asset names the app understands
RAW_CSV_CANDIDATES = ["raw_complaints.csv", "complaints-2026-04-17_04_15_trimmed.csv"]


# --------------------------------------------------------------------------- #
# File resolution (uploaded files take priority over files next to app.py)
# --------------------------------------------------------------------------- #
def get_hf_model_id():
    """Optional: load the model straight from the Hugging Face Hub.

    Resolution order:
      1. Streamlit secret `HF_MODEL_ID` (Settings -> Secrets)
      2. Environment variable `HF_MODEL_ID`
      3. Built-in default: this project's published model on the Hub
    """
    try:
        if "HF_MODEL_ID" in st.secrets:
            return st.secrets["HF_MODEL_ID"]
    except Exception:
        pass
    return os.environ.get("HF_MODEL_ID") or "Mkashif23/cfpb-distilbert"


def resolve(name: str):
    """Return a usable path for `name`.

    Lookup order:
      1. A file sitting next to app.py (e.g. committed to GitHub).
      2. The same filename inside the Hugging Face repo `HF_MODEL_ID`
         (downloaded once, then cached).
      3. None if neither has it.
    """
    local = BASE_DIR / name
    if local.exists():
        return local
    hf_id = get_hf_model_id()
    if hf_id:
        p = _hf_fetch(hf_id, name)
        if p is not None:
            return Path(p)
    return None


@st.cache_data(show_spinner=False)
def _hf_fetch(repo_id: str, filename: str):
    """Download `filename` from the HF repo and return a local path, or None."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return None


def resolve_any(names):
    for n in names:
        p = resolve(n)
        if p is not None:
            return p
    return None


def file_sig(path):
    """Signature used to bust caches when a file changes."""
    if path is None:
        return None
    s = path.stat()
    return (str(path), s.st_size, int(s.st_mtime))


# --------------------------------------------------------------------------- #
# Cached loaders (keyed on file signature so re-uploads refresh automatically)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _read_json(sig):
    with open(sig[0], "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_json(name: str):
    p = resolve(name)
    return _read_json(file_sig(p)) if p else None


@st.cache_data(show_spinner="Reading split distribution…")
def _read_products(sig):
    df = pd.read_csv(sig[0], usecols=["Product"])
    return df["Product"].value_counts()


def load_split_products(name: str):
    p = resolve(name)
    if not p:
        return None
    try:
        return _read_products(file_sig(p))
    except Exception:
        return None


@st.cache_data(show_spinner="Loading sample rows…")
def _read_sample(sig, n):
    return pd.read_csv(sig[0], nrows=n)


def load_split_sample(name: str, n: int = 3):
    p = resolve(name)
    if not p:
        return None
    try:
        return _read_sample(file_sig(p), n)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading raw narratives…")
def _read_raw(sig, n):
    df = pd.read_csv(
        sig[0], usecols=["Product", "Consumer complaint narrative"], nrows=n * 6
    )
    return df.dropna(subset=["Consumer complaint narrative"]).head(n).reset_index(drop=True)


def load_raw_examples(n: int = 6):
    p = resolve_any(RAW_CSV_CANDIDATES)
    if not p:
        return None
    try:
        return _read_raw(file_sig(p), n)
    except Exception:
        return None


# --- Accountability / institutions ------------------------------------------ #
INST_COLS = [
    "Company", "Timely response?", "Consumer disputed?",
    "Product", "Date received", "State",
]


@st.cache_data(show_spinner="Loading complaints data for institutional analysis…")
def _read_for_institutions(sig):
    # Read only what we need; the raw CFPB CSV has 18 columns and is large.
    df = pd.read_csv(sig[0], usecols=lambda c: c in INST_COLS, dtype=str,
                     on_bad_lines="skip")
    df["Company"] = df["Company"].fillna("Unknown").str.strip()
    # CFPB date format is MM/DD/YY
    if "Date received" in df.columns:
        df["Date received"] = pd.to_datetime(
            df["Date received"], format="%m/%d/%y", errors="coerce"
        )
    return df


def load_institutional_data():
    p = resolve_any(RAW_CSV_CANDIDATES)
    if not p:
        return None
    try:
        return _read_for_institutions(file_sig(p))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def compute_institution_risk(_df_key, min_complaints: int = 30):
    """Composite risk score per institution.

    Score combines three percentile-ranked signals with equal weight:
      - Complaint volume (higher = riskier)
      - Untimely-response rate
      - Consumer-dispute rate
    """
    # _df_key is just a cache token; pull the data fresh
    df = load_institutional_data()
    if df is None or "Company" not in df.columns:
        return None

    def _rate(series, target):
        s = series.dropna().astype(str).str.strip().str.lower()
        return (s == target).mean() if len(s) else np.nan

    g = df.groupby("Company", dropna=False)
    stats = pd.DataFrame({
        "complaints": g.size(),
    })
    if "Timely response?" in df.columns:
        stats["untimely_rate"] = g["Timely response?"].apply(lambda s: _rate(s, "no"))
    else:
        stats["untimely_rate"] = np.nan
    if "Consumer disputed?" in df.columns:
        stats["dispute_rate"] = g["Consumer disputed?"].apply(lambda s: _rate(s, "yes"))
    else:
        stats["dispute_rate"] = np.nan

    stats = stats.reset_index()
    stats = stats[stats["complaints"] >= min_complaints].copy()
    if stats.empty:
        return stats

    # Fill missing rates with median so companies aren't penalised for missing data
    for col in ["untimely_rate", "dispute_rate"]:
        med = stats[col].median()
        if pd.notna(med):
            stats[col] = stats[col].fillna(med)
        else:
            stats[col] = stats[col].fillna(0.0)

    # Percentile-rank each signal (higher percentile = higher risk)
    stats["volume_pct"] = stats["complaints"].rank(pct=True)
    stats["untimely_pct"] = stats["untimely_rate"].rank(pct=True)
    stats["dispute_pct"] = stats["dispute_rate"].rank(pct=True)
    stats["composite"] = (
        stats["volume_pct"] + stats["untimely_pct"] + stats["dispute_pct"]
    ) / 3.0
    return stats.sort_values("composite", ascending=False).reset_index(drop=True)



def compute_metrics(cm: np.ndarray):
    support = cm.sum(axis=1)
    pred_tot = cm.sum(axis=0)
    diag = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_tot > 0, diag / pred_tot, 0.0)
        recall = np.where(support > 0, diag / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    total = cm.sum()
    accuracy = diag.sum() / total if total else 0.0
    per_class = pd.DataFrame(
        {
            "Category": LABELS_ORDERED,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support.astype(int),
        }
    )
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / total if total else 0.0
    return per_class, accuracy, macro_f1, weighted_f1


# --------------------------------------------------------------------------- #
# Model loading for the live classifier
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model from Hugging Face Hub…")
def load_model_hf(repo_id: str):
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()
    return tok, model


# --------------------------------------------------------------------------- #
# Sidebar: uploader + navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🏦 Complaint Classifier")
st.sidebar.caption("Automated classification of consumer financial complaints (NLP)")

# Show where the live model is loaded from
_hf_id = get_hf_model_id()
if _hf_id:
    st.sidebar.markdown(
        f"<span class='muted'>Model source:<br>🤗 <code>{_hf_id}</code></span>",
        unsafe_allow_html=True,
    )

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Dataset", "Training", "Evaluation",
     "Live classifier", "Accountability", "Model card"],
)

per_class_df, ACCURACY, MACRO_F1, WEIGHTED_F1 = compute_metrics(CONFUSION_MATRIX)
config_json = load_json("config.json")
meta_json = load_json("pipeline_metadata.json")
tok_cfg_json = load_json("tokenizer_config.json")

st.sidebar.markdown("---")
st.sidebar.metric("Test accuracy", f"{ACCURACY*100:.1f}%")
st.sidebar.metric("Macro F1", f"{MACRO_F1:.3f}")
st.sidebar.caption(f"{int(CONFUSION_MATRIX.sum()):,} test samples · 11 classes")


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
if page == "Overview":
    st.title("Automated Classification of Consumer Financial Complaints")
    st.markdown(
        "A fine-tuned **DistilBERT** model that reads a consumer's complaint "
        "narrative and routes it to one of **11 financial-product categories** — "
        "the kind of triage a CFPB intake team would otherwise do by hand."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{ACCURACY*100:.1f}%")
    c2.metric("Macro F1", f"{MACRO_F1:.3f}")
    c3.metric("Weighted F1", f"{WEIGHTED_F1:.3f}")
    c4.metric("Classes", "11")

    st.markdown("### How it works")
    st.markdown(
        "1. **Input** — a free-text complaint narrative.\n"
        "2. **Tokenize** — DistilBERT WordPiece tokenizer, max length 512, "
        "with PII masked to a `[REDACTED]` token.\n"
        "3. **Classify** — a 6-layer transformer encoder + linear head outputs "
        "a probability over the 11 product categories.\n"
        "4. **Output** — the predicted product and the model's confidence."
    )

    st.markdown("### Categories")
    st.markdown("".join(f"<span class='pill'>{l}</span>" for l in LABELS_ORDERED),
                unsafe_allow_html=True)

    st.markdown("### Best / worst performing categories")
    ranked = per_class_df.sort_values("F1", ascending=False)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Strongest (F1)**")
        st.dataframe(ranked.head(3)[["Category", "F1", "Support"]],
                     hide_index=True, width="stretch")
    with cc2:
        st.markdown("**Weakest (F1)**")
        st.dataframe(ranked.tail(3)[["Category", "F1", "Support"]],
                     hide_index=True, width="stretch")

    if not any_detected:
        st.info("Upload your project files from the sidebar to populate every "
                "page. The Overview and Evaluation pages work out of the box.",
                icon="📤")


# --------------------------------------------------------------------------- #
# Page: Dataset
# --------------------------------------------------------------------------- #
elif page == "Dataset":
    st.title("📊 Dataset")
    st.markdown(
        "Source: CFPB consumer-complaint database. Narratives were cleaned, "
        "PII-redacted, tokenized, and split into train / validation / test sets."
    )

    splits = {
        "Train": "processed_train.csv",
        "Validation": "processed_val.csv",
        "Test": "processed_test.csv",
    }
    dists = {name: load_split_products(fname) for name, fname in splits.items()}

    cols = st.columns(3)
    for col, (name, dist) in zip(cols, dists.items()):
        col.metric(f"{name} samples", f"{int(dist.sum()):,}" if dist is not None else "—")

    st.markdown("### Class distribution")
    avail = {k: v for k, v in dists.items() if v is not None}
    if avail:
        frames = []
        for name, dist in avail.items():
            d = dist.rename_axis("Product").reset_index(name="Count")
            d["Split"] = name
            frames.append(d)
        plot_df = pd.concat(frames, ignore_index=True)
        order = plot_df.groupby("Product")["Count"].sum().sort_values().index
        fig = px.bar(plot_df, x="Count", y="Product", color="Split",
                     orientation="h", barmode="group",
                     category_orders={"Product": list(order)}, height=480)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "<span class='muted'>The dataset is heavily imbalanced — "
            "credit-reporting complaints dwarf every other category — which is "
            "why class weights are used during training.</span>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload the processed train / val / test CSVs from the sidebar "
                "to see class distributions here.", icon="📤")

    st.markdown("### Class weights used in training")
    cw = (pd.DataFrame({"Category": list(CLASS_WEIGHTS.keys()),
                        "Weight": list(CLASS_WEIGHTS.values())})
          .sort_values("Weight", ascending=False))
    st.dataframe(cw, hide_index=True, width="stretch",
                 column_config={"Weight": st.column_config.NumberColumn(format="%.3f")})
    st.markdown(
        "<span class='muted'>Rare classes (e.g. Debt/credit management, Prepaid "
        "card) get large weights so the loss does not ignore them.</span>",
        unsafe_allow_html=True,
    )

    st.markdown("### What a tokenized row looks like")
    sample = load_split_sample("processed_train.csv", n=3)
    if sample is not None:
        st.caption(
            "The narrative column stores DistilBERT input IDs (101 = [CLS], "
            f"102 = [SEP], 0 = [PAD], {REDACTED_TOKEN_ID} = [REDACTED]), padded to 512."
        )
        st.dataframe(sample, width="stretch", height=180)
    else:
        st.caption("Bundle processed_train.csv with the app to preview a tokenized row.")

    st.markdown("### Raw complaint examples")
    raw = load_raw_examples(n=6)
    if raw is not None:
        for _, row in raw.iterrows():
            with st.expander(f"📄 {row['Product']}"):
                txt = str(row["Consumer complaint narrative"])
                st.write(txt[:1200] + ("…" if len(txt) > 1200 else ""))
    else:
        st.caption("Bundle a raw complaints CSV to show example narratives here.")


# --------------------------------------------------------------------------- #
# Page: Training
# --------------------------------------------------------------------------- #
elif page == "Training":
    st.title("📈 Training")
    img = resolve("training_curves.png")
    if img is not None:
        st.image(str(img), width="stretch",
                 caption="Loss and validation macro-F1 over epochs")
    else:
        st.info("Upload training_curves.png from the sidebar to display the "
                "training curves.", icon="📤")

    st.markdown("### Reading the curves")
    st.markdown(
        "- **Training loss** falls steadily (≈1.10 → 0.54), so the model is "
        "learning the task.\n"
        "- **Validation loss** dips at epoch 1 and then ticks back up — the "
        "classic early sign of mild **overfitting** beyond ~1 epoch.\n"
        "- **Validation macro-F1** still climbs across all three epochs "
        "(≈0.68 → 0.72 → 0.73), so generalization on the metric we care about "
        "keeps improving.\n\n"
        "Takeaway: keep the checkpoint with the highest validation macro-F1; "
        "consider early stopping / more regularization beyond 3 epochs."
    )

    cfg = config_json or {}
    st.markdown("### Model setup")
    c1, c2, c3 = st.columns(3)
    c1.metric("Architecture", cfg.get("model_type", "distilbert"))
    c2.metric("Layers", cfg.get("n_layers", 6))
    c3.metric("Hidden dim", cfg.get("dim", 768))
    c1.metric("Attention heads", cfg.get("n_heads", 12))
    c2.metric("Max sequence len", cfg.get("max_position_embeddings", 512))
    c3.metric("Vocab size", f"{cfg.get('vocab_size', 30523):,}")


# --------------------------------------------------------------------------- #
# Page: Evaluation
# --------------------------------------------------------------------------- #
elif page == "Evaluation":
    st.title("🎯 Evaluation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{ACCURACY*100:.1f}%")
    c2.metric("Macro F1", f"{MACRO_F1:.3f}")
    c3.metric("Weighted F1", f"{WEIGHTED_F1:.3f}")
    c4.metric("Test samples", f"{int(CONFUSION_MATRIX.sum()):,}")

    tab1, tab2, tab3 = st.tabs(["Per-class metrics", "Confusion matrix", "Top confusions"])

    with tab1:
        st.markdown("Computed live from the confusion matrix.")
        st.dataframe(
            per_class_df, hide_index=True, width="stretch",
            column_config={
                "Precision": st.column_config.ProgressColumn(
                    "Precision", min_value=0.0, max_value=1.0, format="%.3f"),
                "Recall": st.column_config.ProgressColumn(
                    "Recall", min_value=0.0, max_value=1.0, format="%.3f"),
                "F1": st.column_config.ProgressColumn(
                    "F1", min_value=0.0, max_value=1.0, format="%.3f"),
            },
        )
        fig = px.bar(per_class_df.assign(Short=SHORT_LABELS).sort_values("F1"),
                     x="F1", y="Short", orientation="h", range_x=[0, 1], height=420)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="", xaxis_title="F1 score")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        normalize = st.toggle("Normalize by true class (row %)", value=False)
        if normalize:
            row_sums = CONFUSION_MATRIX.sum(axis=1, keepdims=True)
            mat = np.divide(CONFUSION_MATRIX, row_sums,
                            out=np.zeros_like(CONFUSION_MATRIX), where=row_sums > 0)
            text = [[f"{v*100:.0f}%" for v in r] for r in mat]
            colorbar_title = "Row %"
        else:
            mat = CONFUSION_MATRIX
            text = [[f"{int(v)}" for v in r] for r in mat]
            colorbar_title = "Count"
        fig = go.Figure(data=go.Heatmap(
            z=mat, x=SHORT_LABELS, y=SHORT_LABELS, text=text,
            texttemplate="%{text}", colorscale="Blues",
            colorbar=dict(title=colorbar_title)))
        fig.update_layout(height=620, xaxis_title="Predicted", yaxis_title="True",
                          yaxis_autorange="reversed", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        if resolve("confusion_matrix.png") is not None:
            st.caption("A static reference image (confusion_matrix.png) is also "
                       "available in your files.")

    with tab3:
        st.markdown("Largest off-diagonal cells — where the model gets confused.")
        rows = []
        n = CONFUSION_MATRIX.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and CONFUSION_MATRIX[i, j] > 0:
                    rows.append({
                        "True": LABELS_ORDERED[i],
                        "Predicted as": LABELS_ORDERED[j],
                        "Count": int(CONFUSION_MATRIX[i, j]),
                        "% of true class": CONFUSION_MATRIX[i, j] / CONFUSION_MATRIX[i].sum(),
                    })
        conf_df = pd.DataFrame(rows).sort_values("Count", ascending=False).head(12)
        st.dataframe(conf_df, hide_index=True, width="stretch",
                     column_config={"% of true class":
                                    st.column_config.NumberColumn(format="%.1f%%")})
        st.markdown(
            "<span class='muted'>The biggest leakage is Debt collection and "
            "Vehicle/Student loans being absorbed into the dominant "
            "Credit-reporting class — typical when one category overwhelms the "
            "data.</span>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page: Live classifier
# --------------------------------------------------------------------------- #
elif page == "Live classifier":
    st.title("🤖 Live classifier")
    st.markdown(
        "Type or paste a complaint narrative and the model will predict the "
        "financial-product category."
    )

    examples = {
        "— pick an example —": "",
        "Credit card dispute":
            "I was charged twice for the same purchase on my credit card and the "
            "company refuses to reverse the duplicate charge after several calls.",
        "Mortgage servicing":
            "My mortgage servicer applied my extra payment to interest instead of "
            "principal and now my escrow balance is wrong.",
        "Debt collection":
            "A debt collector keeps calling me about an account I already paid off "
            "and they are reporting it as past due on my credit file.",
    }
    pick = st.selectbox("Quick examples", list(examples.keys()))
    text = st.text_area("Complaint narrative", value=examples[pick], height=160,
                        placeholder="Describe the complaint here…")

    hf_id = get_hf_model_id()
    st.caption(f"Model source: 🤗 Hugging Face Hub — `{hf_id}`")

    if st.button("Classify", type="primary", disabled=not text.strip()):
        try:
            import torch
            import torch.nn.functional as F

            tok, model = load_model_hf(hf_id)
            with st.spinner("Running the model…"):
                inputs = tok(text, truncation=True, max_length=512,
                             return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1).squeeze().tolist()

            pred_id = int(np.argmax(probs))
            st.success(f"**Predicted category:** {ID2LABEL[pred_id]}")
            st.metric("Confidence", f"{probs[pred_id]*100:.1f}%")
            prob_df = (pd.DataFrame({"Category": SHORT_LABELS,
                                     "Probability": probs})
                       .sort_values("Probability"))
            fig = px.bar(prob_df, x="Probability", y="Category",
                         orientation="h", range_x=[0, 1], height=420)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        except ModuleNotFoundError:
            st.error("PyTorch / Transformers not installed. Run "
                     "`pip install torch transformers safetensors`.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not run the model: {exc}")


# --------------------------------------------------------------------------- #
# Page: Accountability / High-Risk Institutions  (SDG 16 framing)
# --------------------------------------------------------------------------- #
elif page == "Accountability":
    st.title("🏛️ Accountability & High-Risk Institutions")
    st.markdown(
        "Institution-level view of the complaints record, intended to support "
        "**SDG 16** (peace, justice and strong institutions) by surfacing where "
        "the burden of consumer complaints is concentrated and which companies "
        "show systemic weaknesses in their response."
    )

    inst_df = load_institutional_data()
    if inst_df is None or inst_df.empty:
        st.info(
            "Bundle a raw CFPB complaints CSV alongside app.py (any file with "
            "`complaint` in the name and the standard columns: Company, "
            "`Timely response?`, `Consumer disputed?`, Product, Date received). "
            "Without it, this page can't compute institution-level risk.",
            icon="📤",
        )
    else:
        # ---- controls --------------------------------------------------------
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            min_complaints = st.slider(
                "Minimum complaints to qualify", 5, 500, 30, step=5,
                help="Companies with very few complaints are excluded so a single "
                     "dispute doesn't dominate the score.",
            )
        with c2:
            threshold = st.slider(
                "High-risk composite threshold", 0.50, 0.99, 0.75, step=0.01,
                help="Companies with a composite score above this value are flagged "
                     "as high-risk. Lower the threshold if the dispute-rate field is "
                     "absent (CFPB stopped collecting it in recent years).",
            )
        with c3:
            top_n = st.slider("How many institutions to chart", 5, 40, 15, step=1)

        risk = compute_institution_risk(file_sig(resolve_any(RAW_CSV_CANDIDATES)),
                                        min_complaints=min_complaints)
        if risk is None or risk.empty:
            st.warning("No companies meet the minimum complaint threshold. Lower it to see results.")
        else:
            risk["high_risk"] = risk["composite"] >= threshold

            # ---- KPI row -----------------------------------------------------
            tot_complaints = int(inst_df.shape[0])
            tot_companies = int(risk.shape[0])
            high_risk_n = int(risk["high_risk"].sum())
            high_risk_share = (
                risk.loc[risk["high_risk"], "complaints"].sum() / tot_complaints
                if tot_complaints else 0.0
            )
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Complaints analysed", f"{tot_complaints:,}")
            k2.metric("Institutions scored", f"{tot_companies:,}")
            k3.metric("Flagged high-risk", f"{high_risk_n:,}")
            k4.metric("Their share of all complaints", f"{high_risk_share*100:.1f}%")

            # ---- top-N composite chart --------------------------------------
            st.markdown("### Top institutions by composite risk score")
            top = risk.head(top_n).copy()
            top["Status"] = np.where(top["high_risk"], "High-risk", "Elevated")
            fig = px.bar(
                top.iloc[::-1],  # so the highest is at the top of the bar chart
                x="composite", y="Company", color="Status",
                orientation="h", range_x=[0, 1],
                color_discrete_map={"High-risk": "#DC2626", "Elevated": "#0E7490"},
                hover_data={"complaints": True, "untimely_rate": ":.1%",
                            "dispute_rate": ":.1%", "composite": ":.3f", "Status": False},
                height=max(360, 24 * len(top)),
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title="", xaxis_title="Composite risk score (0–1)",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, width="stretch")

            # ---- detailed table ---------------------------------------------
            st.markdown("### Institution risk table")
            show = risk.copy()
            show["Flag"] = np.where(show["high_risk"], "🚩 High-risk", "—")
            show = show.rename(columns={
                "Company": "Company",
                "complaints": "Complaints",
                "untimely_rate": "Untimely response",
                "dispute_rate": "Consumer dispute",
                "composite": "Composite",
            })[["Company", "Complaints", "Untimely response",
                "Consumer dispute", "Composite", "Flag"]]
            st.dataframe(
                show, hide_index=True, width="stretch", height=420,
                column_config={
                    "Complaints": st.column_config.NumberColumn(format="%d"),
                    "Untimely response": st.column_config.ProgressColumn(
                        "Untimely response", min_value=0.0, max_value=1.0, format="%.1f%%"),
                    "Consumer dispute": st.column_config.ProgressColumn(
                        "Consumer dispute", min_value=0.0, max_value=1.0, format="%.1f%%"),
                    "Composite": st.column_config.ProgressColumn(
                        "Composite", min_value=0.0, max_value=1.0, format="%.3f"),
                },
            )

            # ---- systemic patterns -------------------------------------------
            st.markdown("### Systemic patterns — complaint mix for top institutions")
            top_companies = risk.head(top_n)["Company"].tolist()
            sub = inst_df[inst_df["Company"].isin(top_companies)]
            if not sub.empty and "Product" in sub.columns:
                mix = (sub.groupby(["Company", "Product"]).size()
                       .reset_index(name="Count"))
                fig2 = px.bar(
                    mix, x="Count", y="Company", color="Product",
                    orientation="h", height=max(360, 24 * len(top_companies)),
                    category_orders={"Company": list(reversed(top_companies))},
                )
                fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                   yaxis_title="", xaxis_title="Number of complaints",
                                   legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig2, width="stretch")
            else:
                st.caption("Product breakdown not available in the uploaded data.")

            # ---- trend over time --------------------------------------------
            if "Date received" in inst_df.columns:
                st.markdown("### Complaints over time — top 5 high-risk institutions")
                top5 = risk.head(5)["Company"].tolist()
                trend_src = inst_df[inst_df["Company"].isin(top5) &
                                    inst_df["Date received"].notna()].copy()
                if not trend_src.empty:
                    trend_src["Month"] = trend_src["Date received"].dt.to_period("M").dt.to_timestamp()
                    trend = (trend_src.groupby(["Month", "Company"]).size()
                             .reset_index(name="Complaints"))
                    fig3 = px.line(trend, x="Month", y="Complaints", color="Company",
                                   markers=True, height=380)
                    fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                       legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(fig3, width="stretch")
                else:
                    st.caption("No usable dates in the uploaded data.")

            # ---- methodology / SDG note -------------------------------------
            with st.expander("How the composite score is computed"):
                st.markdown(
                    "For each company with at least the chosen minimum number of "
                    "complaints, three signals are calculated: **complaint volume**, "
                    "**untimely-response rate** (share of complaints where the company "
                    "did not respond on time), and **consumer-dispute rate** (share of "
                    "complaints the consumer disputed after the company's response). "
                    "Each signal is converted to a **percentile rank across companies "
                    "(0–1)**, then the three percentiles are averaged with equal weight "
                    "to give a composite risk score. Companies above the chosen "
                    "threshold are flagged as **high-risk**. Missing values in either "
                    "rate are imputed with the median across institutions so a company "
                    "isn't penalised for absent data.\n\n"
                    "**Caveat:** the CFPB stopped collecting the *Consumer disputed?* "
                    "field for newer complaints, so that signal is only fully present "
                    "for older records. A high composite score is a screening signal "
                    "for further investigation, not a finding of wrongdoing."
                )

            st.markdown(
                "<span class='muted'>This view supports SDG 16 by making the "
                "complaint footprint of major financial institutions visible and "
                "comparable, contributing to transparency and accountability.</span>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Page: Model card
# --------------------------------------------------------------------------- #
elif page == "Model card":
    st.title("📋 Model card")
    cfg = config_json or {}
    st.markdown("### Summary")
    st.markdown(
        f"- **Task:** multi-class text classification (11 financial products)\n"
        f"- **Base model:** `{(meta_json or {}).get('tokenizer_config', 'distilbert-base-uncased')}`\n"
        f"- **Architecture:** {', '.join(cfg.get('architectures', ['DistilBertForSequenceClassification']))}\n"
        f"- **Parameters:** ~66M\n"
        f"- **Max sequence length:** {cfg.get('max_position_embeddings', 512)}\n"
        f"- **PII handling:** account numbers, dates and names masked to a "
        f"`[REDACTED]` special token"
    )
    st.markdown("### Label mapping")
    lm_df = pd.DataFrame({"ID": list(LABEL_MAPPING.values()),
                          "Category": list(LABEL_MAPPING.keys())}).sort_values("ID")
    st.dataframe(lm_df, hide_index=True, width="stretch")
    with st.expander("Raw config.json"):
        st.json(cfg if cfg else {"note": "config.json not uploaded"})
    with st.expander("Raw pipeline_metadata.json"):
        st.json(meta_json if meta_json else {"note": "pipeline_metadata.json not uploaded"})
    with st.expander("Raw tokenizer_config.json"):
        st.json(tok_cfg_json if tok_cfg_json else {"note": "tokenizer_config.json not uploaded"})
    st.markdown("### Intended use & limitations")
    st.markdown(
        "- **Intended use:** triage / routing of incoming consumer complaints.\n"
        "- **Limitation:** strong bias toward the majority *Credit reporting* "
        "class; rare classes have low support and weaker recall.\n"
        "- **Not for:** legal, financial, or eligibility decisions about an "
        "individual — it predicts a product category, nothing more."
    )

st.sidebar.markdown("---")
st.sidebar.caption("DistilBERT · 11-class CFPB complaint classifier")
