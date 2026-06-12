"""
Automated Classification of Consumer Financial Complaints using NLP
-------------------------------------------------------------------
Streamlit dashboard for a fine-tuned DistilBERT model that classifies
CFPB consumer-complaint narratives into 11 financial-product categories.

The live classifier mirrors the training pipeline exactly (adaptive PII
masking -> [REDACTED], token-aware Head+Tail truncation) and shows an
attention-based token attribution. The Accountability page implements
Research Objective 2 (SDG 16): a composite high-risk institution score,
systemic complaint-pattern charts, and a model-in-the-loop check that
runs the fine-tuned model over live narratives.

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
import re
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

        /* ---------- Landing page ---------- */
        .hero {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b6fff 55%, #06b6d4 100%);
            border-radius: 22px;
            padding: 46px 44px 40px 44px;
            color: #ffffff;
            margin-bottom: 26px;
            box-shadow: 0 18px 45px rgba(30,58,138,0.30);
        }
        .hero h1 {
            color:#ffffff; font-size: 2.35rem; line-height: 1.15;
            margin: 0 0 10px 0; letter-spacing: -1px;
        }
        .hero p {font-size: 1.05rem; opacity: 0.92; max-width: 720px; margin: 0;}
        .hero-badge {
            display:inline-block; padding:5px 14px; margin:0 8px 14px 0;
            border-radius:999px; font-size:0.78rem; font-weight:600;
            letter-spacing:0.4px; text-transform:uppercase;
            background:rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.35);
            color:#ffffff;
        }
        .sdg-badge {background:#f59e0b; border-color:#f59e0b; color:#1f2937;}
        .feature-card {
            background:#ffffff;
            border:1px solid rgba(130,150,200,0.25);
            border-radius:16px; padding:20px 20px 16px 20px;
            height:100%; min-height:172px;
            box-shadow:0 4px 14px rgba(30,58,138,0.06);
            transition: transform .15s ease;
        }
        .feature-card:hover {transform: translateY(-3px);}
        .feature-card .fc-icon {font-size:1.7rem;}
        .feature-card h4 {margin:6px 0 6px 0; font-size:1.02rem;}
        .feature-card p {font-size:0.86rem; opacity:0.75; margin:0;}
        .stat-card {
            background: linear-gradient(160deg, rgba(59,111,255,0.10), rgba(6,182,212,0.10));
            border:1px solid rgba(59,111,255,0.25);
            border-radius:16px; padding:18px 20px; text-align:center;
        }
        .stat-card .num {font-size:1.9rem; font-weight:700; color:#1e3a8a;}
        .stat-card .lbl {font-size:0.82rem; opacity:0.7;}
        .team-chip {
            display:inline-block; padding:6px 14px; margin:3px 6px 3px 0;
            border-radius:999px; font-size:0.84rem;
            background:rgba(30,58,138,0.07); border:1px solid rgba(30,58,138,0.18);
        }
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


@st.cache_data(show_spinner="Sampling narratives for the model-in-the-loop check…")
def _read_narrative_sample(sig, n):
    cols = ["Company", "Product", "Consumer complaint narrative"]
    df = pd.read_csv(sig[0], usecols=lambda c: c in cols, dtype=str,
                     on_bad_lines="skip", nrows=20000)
    df = df.dropna(subset=["Consumer complaint narrative", "Product"])
    df["Company"] = df["Company"].fillna("Unknown").str.strip()
    return df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)


def load_narrative_sample(n: int = 60):
    p = resolve_any(RAW_CSV_CANDIDATES)
    if not p:
        return None
    try:
        return _read_narrative_sample(file_sig(p), n)
    except Exception:
        return None


def compute_institution_risk(df, min_complaints: int = 30):
    """Composite risk score per institution.

    Score combines three percentile-ranked signals with equal weight:
      - Complaint volume (higher = riskier)
      - Untimely-response rate
      - Consumer-dispute rate
    """
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
# Inference-time preprocessing — mirrors the training pipeline exactly
# (adaptive PII masking -> [REDACTED], Head+Tail 255/255 truncation)
# --------------------------------------------------------------------------- #
_RE_XMASK = re.compile(r"(?<![A-Za-z0-9])X{2,}[\dX/.-]*(?![A-Za-z])")
_RE_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")


def clean_text_adaptive(text: str) -> str:
    """Replicate the training-time cleaning: CFPB 'XXXX' PII masks and
    explicit dates are mapped to the single special token [REDACTED]."""
    text = _RE_XMASK.sub(" [REDACTED] ", text)
    text = _RE_DATE.sub(" [REDACTED] ", text)
    return re.sub(r"\s+", " ", text).strip()


def head_tail_encode(tok, text: str, max_len: int = 512,
                     head: int = 255, tail: int = 255):
    """Token-aware Head+Tail truncation (proposal §3.2).

    If the narrative exceeds the 512-token limit, keep the first 255 and
    last 255 tokens so both the framing and the resolution survive,
    instead of plain right-truncation.
    """
    import torch
    ids = tok(text, add_special_tokens=False)["input_ids"]
    truncated = len(ids) + 2 > max_len
    body = ids[:head] + ids[-tail:] if truncated else ids
    input_ids = [tok.cls_token_id] + body + [tok.sep_token_id]
    attn = [1] * len(input_ids)
    return (torch.tensor([input_ids]), torch.tensor([attn]),
            truncated, len(ids))


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
    ["🏠 Home", "Live classifier", "Overview", "Dataset", "Training",
     "Evaluation", "Accountability", "Model card"],
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
# Page: Home (landing)
# --------------------------------------------------------------------------- #
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">WQF7007 · Natural Language Processing</span>
            <span class="hero-badge">Group 27 · OCC3</span>
            <span class="hero-badge sdg-badge">🕊️ SDG 16 · Peace, Justice &amp; Strong Institutions</span>
            <h1>Automated Classification of<br>Consumer Financial Complaints</h1>
            <p>A fine-tuned <b>DistilBERT</b> system that reads raw consumer
            complaint narratives, routes them to the right financial-product
            category in real time, and turns the complaints record into an
            <b>accountability dashboard</b> that surfaces high-risk
            institutions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- headline stats ----------------------------------------------------
    s1, s2, s3, s4 = st.columns(4)
    for col, num, lbl in [
        (s1, f"{ACCURACY*100:.1f}%", "Test accuracy"),
        (s2, f"{MACRO_F1:.3f}", "Macro F1 (imbalance-aware)"),
        (s3, "11", "Product categories"),
        (s4, "116,729", "Unique CFPB complaints"),
    ]:
        col.markdown(
            f"<div class='stat-card'><div class='num'>{num}</div>"
            f"<div class='lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ---- what's inside -------------------------------------------------------
    st.markdown("### ✨ What's inside")
    r1 = st.columns(3)
    cards_top = [
        ("🤖", "Live classifier",
         "Paste any complaint narrative and watch the fine-tuned DistilBERT "
         "predict its category with confidence scores — using the exact "
         "training pipeline (PII masking, Head+Tail truncation)."),
        ("🏛️", "Accountability (SDG 16)",
         "A composite risk score flags high-risk institutions from complaint "
         "volume, untimely responses and consumer disputes — with a "
         "model-in-the-loop check on live narratives."),
        ("🎯", "Evaluation",
         "Per-class precision / recall / F1, an interactive confusion matrix, "
         "and the top confusion pairs — honest about where the model is "
         "strong and where it struggles."),
    ]
    for col, (icon, title, desc) in zip(r1, cards_top):
        col.markdown(
            f"<div class='feature-card'><div class='fc-icon'>{icon}</div>"
            f"<h4>{title}</h4><p>{desc}</p></div>",
            unsafe_allow_html=True,
        )
    st.markdown("&nbsp;", unsafe_allow_html=True)
    r2 = st.columns(3)
    cards_bot = [
        ("📊", "Dataset",
         "CFPB Q4-2025 corpus: class distributions across the stratified "
         "70/15/15 split, the class weights that tame extreme imbalance, and "
         "real tokenized rows."),
        ("📈", "Training",
         "Loss and validation macro-F1 curves across 3 epochs of fine-tuning "
         "with weighted CrossEntropyLoss, AdamW and linear warmup."),
        ("📋", "Model card",
         "Architecture, label mapping, intended use and limitations — "
         "transparency by design, as the proposal's ethics section demands."),
    ]
    for col, (icon, title, desc) in zip(r2, cards_bot):
        col.markdown(
            f"<div class='feature-card'><div class='fc-icon'>{icon}</div>"
            f"<h4>{title}</h4><p>{desc}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ---- how it flows --------------------------------------------------------
    st.markdown("### ⚙️ From raw narrative to accountability — in four steps")
    f1, f2, f3, f4 = st.columns(4)
    flow = [
        (f1, "1️⃣ Clean", "PII masks (`XXXX`, dates) become a single "
                          "`[REDACTED]` special token."),
        (f2, "2️⃣ Truncate smart", "Head+Tail keeps the first 255 + last 255 "
                                   "tokens — framing *and* resolution survive."),
        (f3, "3️⃣ Classify", "6-layer DistilBERT encoder + linear head over "
                             "11 product categories."),
        (f4, "4️⃣ Hold to account", "Predictions feed the institution-level "
                                    "risk view supporting SDG 16."),
    ]
    for col, t, d in flow:
        col.markdown(f"**{t}**\n\n<span class='muted'>{d}</span>",
                     unsafe_allow_html=True)

    st.markdown("---")

    # ---- team ----------------------------------------------------------------
    st.markdown("### 👥 Group 27 — Universiti Malaya")
    st.markdown(
        "<span class='team-chip'>Deng Lingyan · 24235830</span>"
        "<span class='team-chip'>Tan Zhao Thong · 24057408</span>"
        "<span class='team-chip'>Liang Shengbao · 24200736</span>"
        "<span class='team-chip'>Jasmine Fu Ming Yee · 24217068</span>"
        "<span class='team-chip'>Mohammad Kashif Mohiudfin · 25094536</span>",
        unsafe_allow_html=True,
    )
    st.caption("Lecturer: Dr. Mohamed Lubani · WQF7007 Natural Language "
               "Processing · Use the sidebar to explore — start with the "
               "**Live classifier**.")


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
elif page == "Overview":
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
        "2. **Preprocess** — adaptive PII masking to a `[REDACTED]` special "
        "token, then token-aware **Head+Tail truncation** (first 255 + last "
        "255 tokens) for long narratives.\n"
        "3. **Classify** — a 6-layer transformer encoder + linear head outputs "
        "a probability over the 11 product categories.\n"
        "4. **Output** — the predicted product, the model's confidence, and "
        "an accountability link to high-risk institutions in that category."
    )

    st.markdown("### Research objectives → where to see them in this demo")
    st.markdown(
        "| Objective | What it says | Where it is demonstrated |\n"
        "|---|---|---|\n"
        "| **RO1** | Tailored preprocessing: Head+Tail truncation, "
        "`[REDACTED]` special token, stratified 70/15/15 split | "
        "**Dataset** page (splits, class weights, tokenized rows) and the "
        "**Live classifier** (same pipeline applied at inference) |\n"
        "| **RO2** | Integrate the model into an accountability dashboard: "
        "visualize systemic complaint patterns and identify high-risk "
        "institutions (SDG 16) | **Accountability** page (composite risk "
        "score, complaint mix, trends, model-in-the-loop check) and the "
        "**Live classifier**'s accountability link |\n"
        "| **RO3** | Fine-tuned DistilBERT with class-weighted loss "
        "outperforming the baseline on majority *and* minority classes | "
        "**Training** and **Evaluation** pages (per-class F1, macro F1, "
        "confusion matrix) |"
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
                # Same pipeline as training: PII mask -> Head+Tail truncation
                cleaned = clean_text_adaptive(text)
                input_ids, attn_mask, truncated, n_tokens = head_tail_encode(
                    tok, cleaned
                )
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                probs = F.softmax(out.logits, dim=-1).squeeze().tolist()

            pred_id = int(np.argmax(probs))
            st.success(f"**Predicted category:** {ID2LABEL[pred_id]}")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{probs[pred_id]*100:.1f}%")
            m2.metric("Narrative length", f"{n_tokens} tokens")
            if cleaned != text.strip():
                st.caption("🔒 PII patterns (XXXX masks, dates) were mapped to "
                           "the `[REDACTED]` special token before inference — "
                           "same as in training.")
            if truncated:
                st.caption("✂️ Narrative exceeded 512 tokens — **Head+Tail "
                           "truncation** kept the first 255 and last 255 "
                           "tokens so the framing *and* the resolution were "
                           "preserved.")

            prob_df = (pd.DataFrame({"Category": SHORT_LABELS,
                                     "Probability": probs})
                       .sort_values("Probability"))
            fig = px.bar(prob_df, x="Probability", y="Category",
                         orientation="h", range_x=[0, 1], height=420)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="")
            st.plotly_chart(fig, width="stretch")

            # ---- RO2 link: route the prediction into accountability ------
            inst_df = load_institutional_data()
            if inst_df is not None and "Product" in inst_df.columns:
                cat = ID2LABEL[pred_id]
                sub = inst_df[inst_df["Product"] == cat]
                risk = compute_institution_risk(inst_df, min_complaints=30)
                if not sub.empty and risk is not None and not risk.empty:
                    st.markdown(
                        f"### 🏛️ Accountability link — *{cat}*"
                    )
                    st.caption(
                        "The model's prediction routes this complaint into the "
                        "accountability view (RO2): the institutions with the "
                        "largest complaint footprint in this category, with "
                        "their composite risk score."
                    )
                    counts = (sub.groupby("Company").size()
                              .rename("Complaints in category").reset_index())
                    merged = (risk.merge(counts, on="Company")
                              .sort_values("Complaints in category",
                                           ascending=False).head(5))
                    merged["Flag"] = np.where(merged["composite"] >= 0.75,
                                              "🚩 High-risk", "—")
                    st.dataframe(
                        merged.rename(columns={"composite": "Composite risk"})[
                            ["Company", "Complaints in category",
                             "Composite risk", "Flag"]],
                        hide_index=True, width="stretch",
                        column_config={
                            "Composite risk": st.column_config.ProgressColumn(
                                "Composite risk", min_value=0.0,
                                max_value=1.0, format="%.3f"),
                        },
                    )
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
    st.info(
        "**Research Objective 2** — *integrate the model into an "
        "accountability-focused dashboard that visualizes systemic complaint "
        "patterns and identifies high-risk institutions.* This page covers all "
        "three parts: the composite **high-risk score**, the **systemic "
        "pattern** charts below, and the **model-in-the-loop** check that runs "
        "the fine-tuned DistilBERT over live narratives.",
        icon="🎯",
    )

    inst_df = load_institutional_data()

    if inst_df is None or inst_df.empty:
        st.caption("Complaints data not available — bundle the raw complaints "
                   "CSV with the app (or host it on the Hugging Face repo) to "
                   "analyse institution-level risk.")
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

        risk = compute_institution_risk(inst_df, min_complaints=min_complaints)
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

            # ---- model-in-the-loop: RO2 "integrate the model" ----------------
            st.markdown("### 🤖 Model-in-the-loop check")
            st.caption(
                "Run the fine-tuned DistilBERT over a random sample of raw "
                "narratives and compare its predictions with the recorded CFPB "
                "category, broken down by institution. This demonstrates the "
                "model operating *inside* the accountability workflow rather "
                "than as a separate demo."
            )
            mc1, mc2 = st.columns([1, 1])
            with mc1:
                sample_n = st.slider("Narratives to classify", 20, 200, 60,
                                     step=20, key="mil_sample_n")
            with mc2:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                run_mil = st.button("Run model on sample", type="primary",
                                    key="mil_run")
            if run_mil:
                sample = load_narrative_sample(sample_n)
                if sample is None or sample.empty:
                    st.warning("No raw narratives available — bundle the raw "
                               "complaints CSV (or upload one) to enable this.")
                else:
                    try:
                        import torch
                        import torch.nn.functional as F

                        tok, model = load_model_hf(get_hf_model_id())
                        preds, confs = [], []
                        prog = st.progress(0.0, text="Classifying narratives…")
                        for i, narr in enumerate(
                                sample["Consumer complaint narrative"]):
                            cleaned = clean_text_adaptive(str(narr))
                            ids, attn, _, _ = head_tail_encode(tok, cleaned)
                            with torch.no_grad():
                                logits = model(input_ids=ids,
                                               attention_mask=attn).logits
                            p = F.softmax(logits, dim=-1).squeeze()
                            preds.append(ID2LABEL[int(p.argmax())])
                            confs.append(float(p.max()))
                            prog.progress((i + 1) / len(sample))
                        prog.empty()

                        sample = sample.assign(Predicted=preds,
                                               Confidence=confs)
                        agree = (sample["Predicted"] ==
                                 sample["Product"]).mean()
                        a1, a2, a3 = st.columns(3)
                        a1.metric("Narratives classified", f"{len(sample):,}")
                        a2.metric("Agreement with recorded label",
                                  f"{agree*100:.1f}%")
                        a3.metric("Mean confidence",
                                  f"{np.mean(confs)*100:.1f}%")

                        mixp = (sample.groupby("Predicted").size()
                                .reset_index(name="Count")
                                .sort_values("Count"))
                        figp = px.bar(mixp, x="Count", y="Predicted",
                                      orientation="h",
                                      height=max(300, 26 * len(mixp)))
                        figp.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                           yaxis_title="",
                                           xaxis_title="Model-predicted "
                                                       "complaint mix")
                        st.plotly_chart(figp, width="stretch")

                        st.markdown("**Per-institution view (sampled)**")
                        st.dataframe(
                            sample[["Company", "Product", "Predicted",
                                    "Confidence"]],
                            hide_index=True, width="stretch", height=320,
                            column_config={
                                "Confidence": st.column_config.ProgressColumn(
                                    "Confidence", min_value=0.0,
                                    max_value=1.0, format="%.1f%%"),
                            },
                        )
                        st.caption(
                            "Disagreements are not necessarily model errors — "
                            "CFPB categories are self-selected by consumers "
                            "and the taxonomy overlaps (e.g. debt collection "
                            "vs. credit reporting)."
                        )
                    except ModuleNotFoundError:
                        st.error("PyTorch / Transformers not installed — the "
                                 "model-in-the-loop check needs the live-"
                                 "classifier dependencies.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not run the model: {exc}")

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
