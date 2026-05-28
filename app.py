"""
Automated Classification of Consumer Financial Complaints using NLP
-------------------------------------------------------------------
Streamlit dashboard for a fine-tuned DistilBERT model that classifies
CFPB consumer-complaint narratives into 11 financial-product categories.

Run with:
    streamlit run app.py

Place these files next to app.py (all produced by the training pipeline):
    config.json
    tokenizer_config.json
    tokenizer.json
    pipeline_metadata.json
    training_curves.png
    confusion_matrix.png
    processed_train.csv / processed_val.csv / processed_test.csv
Optional (only needed for the live "Classify" tab):
    model.safetensors  (or pytorch_model.bin)  -- the trained weights
    (the tokenizer loads from tokenizer.json, so no separate vocab.txt is needed)
Optional (raw text examples in the Dataset tab):
    complaints-2026-04-17_04_15_trimmed.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths & page config
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent


def asset(name: str) -> Path:
    return BASE_DIR / name


st.set_page_config(
    page_title="Complaint Classifier Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# A little styling for a cleaner look
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

# Short names for compact charts (index = class id)
SHORT_LABELS = [
    "Checking/Savings",
    "Credit card",
    "Credit reporting",
    "Debt collection",
    "Debt/credit mgmt",
    "Money transfer",
    "Mortgage",
    "Payday/Personal loan",
    "Prepaid card",
    "Student loan",
    "Vehicle loan",
]

# Confusion matrix read directly from confusion_matrix.png (rows = true, cols = pred,
# both in class-id order 0..10). Embedding it lets the dashboard recompute every
# metric on the fly without needing the raw prediction arrays.
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

REDACTED_TOKEN_ID = 30522  # the "[REDACTED]" extra special token used to mask PII


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_json(name: str):
    p = asset(name)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner="Reading split distribution…")
def load_split_products(name: str):
    """Read only the 'Product' column (skips the huge tokenized column)."""
    p = asset(name)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, usecols=["Product"])
        return df["Product"].value_counts()
    except Exception:
        return None


@st.cache_data(show_spinner="Loading sample rows…")
def load_split_sample(name: str, n: int = 5):
    p = asset(name)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, nrows=n)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading raw narratives…")
def load_raw_examples(name: str, n: int = 8):
    p = asset(name)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(
            p,
            usecols=["Product", "Consumer complaint narrative"],
            nrows=n * 6,
        )
        df = df.dropna(subset=["Consumer complaint narrative"])
        return df.head(n).reset_index(drop=True)
    except Exception:
        return None


def compute_metrics(cm: np.ndarray):
    """Per-class precision / recall / F1 / support + overall accuracy & F1s."""
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
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / total if total else 0.0

    per_class = pd.DataFrame(
        {
            "Category": LABELS_ORDERED,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support.astype(int),
        }
    )
    return per_class, accuracy, macro_f1, weighted_f1


def weights_available() -> bool:
    return any(asset(f).exists() for f in ("model.safetensors", "pytorch_model.bin"))


@st.cache_resource(show_spinner="Loading model (first run only)…")
def load_model():
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(BASE_DIR))
    model.eval()
    return tok, model


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🏦 Complaint Classifier")
st.sidebar.caption("Automated classification of consumer financial complaints (NLP)")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dataset",
        "Training",
        "Evaluation",
        "Live classifier",
        "Model card",
    ],
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
    st.markdown(
        "".join(f"<span class='pill'>{lbl}</span>" for lbl in LABELS_ORDERED),
        unsafe_allow_html=True,
    )

    st.markdown("### Best / worst performing categories")
    ranked = per_class_df.sort_values("F1", ascending=False)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Strongest (F1)**")
        st.dataframe(
            ranked.head(3)[["Category", "F1", "Support"]],
            hide_index=True,
            use_container_width=True,
        )
    with cc2:
        st.markdown("**Weakest (F1)**")
        st.dataframe(
            ranked.tail(3)[["Category", "F1", "Support"]],
            hide_index=True,
            use_container_width=True,
        )

    st.info(
        "Tip: use the sidebar to explore the **Dataset**, **Training** curves, "
        "detailed **Evaluation**, and a **Live classifier** demo.",
        icon="💡",
    )


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
        order = (
            plot_df.groupby("Product")["Count"].sum().sort_values(ascending=True).index
        )
        fig = px.bar(
            plot_df,
            x="Count",
            y="Product",
            color="Split",
            orientation="h",
            barmode="group",
            category_orders={"Product": list(order)},
            height=480,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "<span class='muted'>The dataset is heavily imbalanced — "
            "credit-reporting complaints dwarf every other category — which is "
            "why class weights are used during training.</span>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "Processed split CSVs not found next to app.py, so distributions "
            "can't be drawn. Add processed_train.csv / processed_val.csv / "
            "processed_test.csv to enable this section."
        )

    st.markdown("### Class weights used in training")
    cw = (
        pd.DataFrame({"Category": list(CLASS_WEIGHTS.keys()),
                      "Weight": list(CLASS_WEIGHTS.values())})
        .sort_values("Weight", ascending=False)
    )
    st.dataframe(
        cw,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Weight": st.column_config.NumberColumn(format="%.3f"),
        },
    )
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
        st.dataframe(sample, use_container_width=True, height=180)
    else:
        st.caption("processed_train.csv not found — skipping tokenized preview.")

    st.markdown("### Raw complaint examples")
    raw = load_raw_examples("complaints-2026-04-17_04_15_trimmed.csv", n=6)
    if raw is not None:
        for _, row in raw.iterrows():
            with st.expander(f"📄 {row['Product']}"):
                txt = str(row["Consumer complaint narrative"])
                st.write(txt[:1200] + ("…" if len(txt) > 1200 else ""))
    else:
        st.caption(
            "Add complaints-2026-04-17_04_15_trimmed.csv to show raw narrative "
            "examples here."
        )


# --------------------------------------------------------------------------- #
# Page: Training
# --------------------------------------------------------------------------- #
elif page == "Training":
    st.title("📈 Training")
    img = asset("training_curves.png")
    if img.exists():
        st.image(str(img), use_container_width=True,
                 caption="Loss and validation macro-F1 over epochs")
    else:
        st.warning("training_curves.png not found next to app.py.")

    st.markdown("### Reading the curves")
    st.markdown(
        "- **Training loss** falls steadily (≈1.10 → 0.54), so the model is "
        "learning the task.\n"
        "- **Validation loss** dips at epoch 1 and then ticks back up — the "
        "classic early sign of mild **overfitting** beyond ~1 epoch.\n"
        "- **Validation macro-F1** still climbs across all three epochs "
        "(≈0.68 → 0.72 → 0.73), so generalization on the metric we care about "
        "keeps improving.\n\n"
        "Takeaway: the best checkpoint is the one with the highest validation "
        "macro-F1; consider early stopping / more regularization if pushing "
        "past 3 epochs."
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

    tab1, tab2, tab3 = st.tabs(
        ["Per-class metrics", "Confusion matrix", "Top confusions"]
    )

    with tab1:
        st.markdown("Computed live from the confusion matrix.")
        st.dataframe(
            per_class_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Precision": st.column_config.ProgressColumn(
                    "Precision", min_value=0.0, max_value=1.0, format="%.3f"),
                "Recall": st.column_config.ProgressColumn(
                    "Recall", min_value=0.0, max_value=1.0, format="%.3f"),
                "F1": st.column_config.ProgressColumn(
                    "F1", min_value=0.0, max_value=1.0, format="%.3f"),
            },
        )
        fig = px.bar(
            per_class_df.assign(Short=SHORT_LABELS).sort_values("F1"),
            x="F1", y="Short", orientation="h", range_x=[0, 1], height=420,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="", xaxis_title="F1 score")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        normalize = st.toggle("Normalize by true class (row %)", value=False)
        if normalize:
            row_sums = CONFUSION_MATRIX.sum(axis=1, keepdims=True)
            mat = np.divide(CONFUSION_MATRIX, row_sums,
                            out=np.zeros_like(CONFUSION_MATRIX),
                            where=row_sums > 0)
            text = [[f"{v*100:.0f}%" for v in r] for r in mat]
            colorbar_title = "Row %"
        else:
            mat = CONFUSION_MATRIX
            text = [[f"{int(v)}" for v in r] for r in mat]
            colorbar_title = "Count"

        fig = go.Figure(
            data=go.Heatmap(
                z=mat,
                x=SHORT_LABELS,
                y=SHORT_LABELS,
                text=text,
                texttemplate="%{text}",
                colorscale="Blues",
                colorbar=dict(title=colorbar_title),
            )
        )
        fig.update_layout(
            height=620,
            xaxis_title="Predicted",
            yaxis_title="True",
            yaxis_autorange="reversed",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Static reference image (confusion_matrix.png) is also available "
            "in the project folder."
        )

    with tab3:
        st.markdown("Largest off-diagonal cells — where the model gets confused.")
        rows = []
        n = CONFUSION_MATRIX.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and CONFUSION_MATRIX[i, j] > 0:
                    rows.append(
                        {
                            "True": LABELS_ORDERED[i],
                            "Predicted as": LABELS_ORDERED[j],
                            "Count": int(CONFUSION_MATRIX[i, j]),
                            "% of true class": CONFUSION_MATRIX[i, j]
                            / CONFUSION_MATRIX[i].sum(),
                        }
                    )
        conf_df = pd.DataFrame(rows).sort_values("Count", ascending=False).head(12)
        st.dataframe(
            conf_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "% of true class": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.markdown(
            "<span class='muted'>The biggest leakage is Debt collection and "
            "Vehicle/Student loans being absorbed into the dominant "
            "Credit-reporting class — typical when one category overwhelms the "
            "data.</span>",
            unsafe_allow_html=True,
        )


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
    default_text = examples[pick]
    text = st.text_area("Complaint narrative", value=default_text, height=160,
                        placeholder="Describe the complaint here…")

    if not weights_available():
        st.warning(
            "**Model weights not found.** This folder has the config and "
            "tokenizer, but not the trained weights, so live prediction is "
            "disabled.\n\nTo enable it, add `model.safetensors` (or "
            "`pytorch_model.bin`) from your training run next to `app.py`, then "
            "install the inference extras:\n\n"
            "```\npip install torch transformers safetensors\n```",
            icon="⚠️",
        )
    else:
        if st.button("Classify", type="primary", disabled=not text.strip()):
            try:
                import torch
                import torch.nn.functional as F

                tok, model = load_model()
                with st.spinner("Running the model…"):
                    inputs = tok(
                        text, truncation=True, max_length=512, return_tensors="pt"
                    )
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    probs = F.softmax(logits, dim=-1).squeeze().tolist()

                pred_id = int(np.argmax(probs))
                st.success(f"**Predicted category:** {ID2LABEL[pred_id]}")
                st.metric("Confidence", f"{probs[pred_id]*100:.1f}%")

                prob_df = (
                    pd.DataFrame({"Category": SHORT_LABELS, "Probability": probs})
                    .sort_values("Probability")
                )
                fig = px.bar(prob_df, x="Probability", y="Category",
                             orientation="h", range_x=[0, 1], height=420)
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            except ModuleNotFoundError:
                st.error(
                    "PyTorch / Transformers are not installed. Run "
                    "`pip install torch transformers` to enable live inference."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not run the model: {exc}")


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
    lm_df = pd.DataFrame(
        {"ID": list(LABEL_MAPPING.values()), "Category": list(LABEL_MAPPING.keys())}
    ).sort_values("ID")
    st.dataframe(lm_df, hide_index=True, use_container_width=True)

    with st.expander("Raw config.json"):
        st.json(cfg if cfg else {"note": "config.json not found"})
    with st.expander("Raw pipeline_metadata.json"):
        st.json(meta_json if meta_json else {"note": "pipeline_metadata.json not found"})
    with st.expander("Raw tokenizer_config.json"):
        st.json(tok_cfg_json if tok_cfg_json else {"note": "tokenizer_config.json not found"})

    st.markdown("### Intended use & limitations")
    st.markdown(
        "- **Intended use:** triage / routing of incoming consumer complaints.\n"
        "- **Limitation:** strong bias toward the majority *Credit reporting* "
        "class; rare classes (Debt/credit management, Prepaid card) have low "
        "support and weaker recall.\n"
        "- **Not for:** legal, financial, or eligibility decisions about an "
        "individual — it predicts a product category, nothing more."
    )

st.sidebar.markdown("---")
st.sidebar.caption("DistilBERT · 11-class CFPB complaint classifier")
