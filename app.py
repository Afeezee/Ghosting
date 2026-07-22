"""
app.py
========
GhostWatch — Ghosting Risk Prediction Interface
Streamlit web application for inference using the trained ghosting_model.pkl

Features
    • Themed landing page with a simple gated login
    • Light / dark interface themes (toggle in the sidebar / login screen)
    • Per-prediction feature-contribution chart computed analytically from the
      linear model coefficients (no runtime SHAP dependency — deterministic)

Run locally:
    streamlit run app.py

Login credentials
    Set them in Streamlit secrets under a [credentials] table, e.g.

        # .streamlit/secrets.toml
        [credentials]
        afeez = "a-strong-password"

    If no secrets are configured the app falls back to a demo account
    (username: demo · password: demo).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
from string import Template

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GhostWatch — Ghosting Risk Predictor",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME SYSTEM ─────────────────────────────────────────────────────────────

PALETTES = {
    "light": {
        "bg": "#F5F6FB", "sidebar_bg": "#FFFFFF", "card_bg": "#FFFFFF",
        "card_border": "#E6E8F2", "text": "#1E2233", "text_muted": "#6B7185",
        "accent1": "#6366F1", "accent2": "#8B5CF6",
        "input_bg": "#FFFFFF", "input_border": "#D9DCEA", "grid": "#E6E8F2",
        "metric_bg": "#F5F6FB", "panel_bg": "#EEF0F8",
        "risk_high_bg": "#FEF2F2", "risk_high_border": "#F5A5A0", "risk_high_text": "#DC2626",
        "risk_low_bg": "#ECFDF5", "risk_low_border": "#86E5B8", "risk_low_text": "#059669",
        "shap_pos": "#E11D48", "shap_neg": "#2563EB",
        "gauge_bg": "#EEF0F8", "gauge_mid": "#FEF3C7",
        "bar_neutral": "#C7CBDB", "shadow": "0 1px 3px rgba(20,22,40,0.06)",
    },
    "dark": {
        "bg": "#0E1017", "sidebar_bg": "#151823", "card_bg": "#181B26",
        "card_border": "#2A2E3D", "text": "#E8EAF4", "text_muted": "#9AA0B8",
        "accent1": "#818CF8", "accent2": "#A78BFA",
        "input_bg": "#1E2230", "input_border": "#343A4D", "grid": "#282C3B",
        "metric_bg": "#1E2230", "panel_bg": "#1A1D28",
        "risk_high_bg": "#2A1618", "risk_high_border": "#7F1D1D", "risk_high_text": "#F87171",
        "risk_low_bg": "#0E2A20", "risk_low_border": "#065F46", "risk_low_text": "#34D399",
        "shap_pos": "#FB7185", "shap_neg": "#60A5FA",
        "gauge_bg": "#1E2230", "gauge_mid": "#3F3620",
        "bar_neutral": "#3A3F52", "shadow": "0 1px 3px rgba(0,0,0,0.4)",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "light"

TH = PALETTES[st.session_state.theme]

_CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Lora:wght@500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: $bg; color: $text; }
[data-testid="stAppViewContainer"] { background-color: $bg; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background-color: $sidebar_bg; border-right: 1px solid $card_border; }
section[data-testid="stSidebar"] * { color: $text; }

h1, h2, h3, h4, h5, h6 { color: $text; }

[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, .stSlider label, label { color: $text !important; }
[data-testid="stMarkdownContainer"] p { color: $text; }
[data-testid="stCaptionContainer"], .stCaption { color: $text_muted !important; }

/* Inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {
    background-color: $input_bg !important; border-color: $input_border !important; color: $text !important;
}
[data-baseweb="input"] input, [data-baseweb="select"] div { color: $text !important; }

/* Buttons */
.stButton > button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"],
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, $accent1 0%, $accent2 100%);
    color: #FFFFFF !important; border: none; border-radius: 9px; font-weight: 600;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.06); }
.stButton > button p, [data-testid="stFormSubmitButton"] button p { color: #FFFFFF !important; }

/* Header */
.gw-header {
    background: linear-gradient(135deg, $accent1 0%, $accent2 100%);
    border-radius: 14px; padding: 2rem 2.4rem 1.6rem; margin-bottom: 1.4rem;
    color: #FFFFFF; box-shadow: $shadow;
}
.gw-header h1 { font-family: 'Lora', serif; font-size: 2rem; font-weight: 600; margin: 0 0 .3rem; color: #FFFFFF; }
.gw-header p { font-size: .92rem; opacity: .92; margin: 0; color: #FFFFFF; }

/* Cards */
.gw-card {
    background: $card_bg; border-radius: 12px; padding: 1.4rem 1.6rem;
    border: 1px solid $card_border; margin-bottom: 1rem; box-shadow: $shadow; color: $text;
}
.gw-card-title {
    font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
    color: $accent1; margin-bottom: .6rem;
}

/* Risk badges */
.risk-high { background: $risk_high_bg; border: 1.5px solid $risk_high_border; border-radius: 10px; padding: 1.2rem 1.4rem; text-align: center; }
.risk-low  { background: $risk_low_bg;  border: 1.5px solid $risk_low_border;  border-radius: 10px; padding: 1.2rem 1.4rem; text-align: center; }
.risk-label { font-family: 'Lora', serif; font-size: 1.5rem; font-weight: 600; margin: 0; }
.risk-high .risk-label { color: $risk_high_text; }
.risk-low  .risk-label { color: $risk_low_text; }
.risk-prob { font-size: .88rem; color: $text_muted; margin-top: .3rem; }

/* Sidebar section labels */
.sidebar-section {
    font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .07em;
    color: $accent1 !important; margin: 1.2rem 0 .4rem; border-bottom: 1px solid $card_border; padding-bottom: .25rem;
}

/* Disclaimer */
.disclaimer { font-size: .78rem; color: $text_muted; background: $panel_bg; border-radius: 8px; padding: .8rem 1rem; border-left: 3px solid $accent1; }

/* Metrics */
.metric-box { background: $metric_bg; border-radius: 10px; padding: .8rem 1rem; border: 1px solid $card_border; text-align: center; }
.metric-val { font-size: 1.5rem; font-weight: 600; color: $text; }
.metric-lbl { font-size: .75rem; color: $text_muted; margin-top: .1rem; }

/* Input summary table */
.gw-table { width: 100%; border-collapse: collapse; font-size: .85rem; }
.gw-table td { padding: .5rem .8rem; border-bottom: 1px solid $card_border; color: $text; }
.gw-table td.k { color: $text_muted; }
.gw-table td.v { text-align: right; font-weight: 600; }

/* Landing / login */
.gw-hero {
    background: linear-gradient(135deg, $accent1 0%, $accent2 100%);
    border-radius: 16px; padding: 2.8rem 2.4rem; color: #fff; text-align: center;
    box-shadow: $shadow; margin-bottom: 1.4rem;
}
.gw-hero h1 { font-family: 'Lora', serif; font-size: 2.7rem; margin: 0 0 .4rem; color: #fff; }
.gw-hero p { font-size: 1.02rem; opacity: .94; margin: 0 auto; max-width: 640px; color: #fff; }
.gw-badge {
    display: inline-block; background: rgba(255,255,255,.18); border-radius: 20px;
    padding: .28rem .85rem; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 1rem;
}

#MainMenu, footer { visibility: hidden; }
</style>
""")

st.markdown(_CSS.safe_substitute(**TH), unsafe_allow_html=True)


def theme_toggle(key: str):
    """Render a light/dark switch; rerun immediately so CSS regenerates."""
    dark = st.toggle("🌙  Dark mode", value=(st.session_state.theme == "dark"), key=key)
    new = "dark" if dark else "light"
    if new != st.session_state.theme:
        st.session_state.theme = new
        st.rerun()


def style_fig(fig, height, right_margin=30):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=right_margin, t=10, b=30),
        font=dict(family="Inter, sans-serif", size=12, color=TH["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=TH["grid"], zeroline=False, color=TH["text_muted"]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", color=TH["text"]),
    )
    return fig

# ─── CACHED LOADERS ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "ghosting_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_importance():
    imp_path = os.path.join(os.path.dirname(__file__), "feature_importance.csv")
    return pd.read_csv(imp_path)

@st.cache_data(show_spinner=False)
def load_background():
    """Training feature rows used as the reference distribution for the
    per-prediction feature-contribution chart."""
    data_path = os.path.join(os.path.dirname(__file__), "ghosting_prediction_dataset.csv")
    bg = pd.read_csv(data_path)
    if "ghosted" in bg.columns:
        bg = bg.drop(columns=["ghosted"])
    return bg

# ─── AUTHENTICATION ───────────────────────────────────────────────────────────

def get_credentials():
    """Return ({username: password}, is_demo). Reads a [credentials] table from
    Streamlit secrets when available, otherwise falls back to a demo account."""
    try:
        raw = st.secrets["credentials"]
        creds = {str(k): str(v) for k, v in dict(raw).items()}
        if creds:
            return creds, False
    except Exception:
        pass
    return {"demo": "demo"}, True


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def render_landing():
    CREDS, is_demo = get_credentials()

    top = st.columns([6, 1.3])
    with top[1]:
        theme_toggle("theme_login")

    st.markdown("""
    <div class="gw-hero">
        <div class="gw-badge">Final-year research project</div>
        <h1>👻 GhostWatch</h1>
        <p>A machine-learning tool for estimating the risk of sudden communication
        cessation — <em>ghosting</em> — in romantic relationships. For research and
        awareness purposes only.</p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">What it does</div>
            Analyses 24 communication, psychological, and behavioural signals to
            estimate the likelihood of unexplained withdrawal from a relationship.
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">How it decides</div>
            Every prediction comes with a per-feature contribution chart, so you can
            see exactly which factors pushed the risk up or down.
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">Model performance</div>
            Logistic Regression · 2,000 observations · 5-fold CV
            AUC <strong>0.953</strong> · F1 <strong>0.817</strong> · Recall <strong>0.885</strong>.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    lc = st.columns([1, 1.35, 1])
    with lc[1]:
        st.markdown('<div class="gw-card"><div class="gw-card-title">Sign in to continue</div>',
                    unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            if username in CREDS and password == CREDS[username]:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")
        if is_demo:
            st.caption("Demo access — username: **demo** · password: **demo**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>Research tool only.</strong> GhostWatch is built on a theoretically
        grounded synthetic dataset. Predictions are probabilistic estimates, not
        judgements about any real individual.
    </div>
    """, unsafe_allow_html=True)


if not st.session_state.authenticated:
    render_landing()
    st.stop()

# ─── LOAD MODEL (post-auth) ───────────────────────────────────────────────────

try:
    artefact      = load_model()
    pipeline      = artefact["pipeline"]
    num_features  = artefact["num_features"]
    ord_features  = artefact["ord_features"]
    nom_features  = artefact["nom_features"]
    best_model    = artefact["best_model"]
    importance_df = load_importance()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Run `model_pipeline.py` first to generate `ghosting_model.pkl`.")
    st.stop()

# ─── SIDEBAR — ACCOUNT, THEME & INPUT FORM ────────────────────────────────────

with st.sidebar:
    who = st.session_state.get("username", "user")
    st.markdown(f"**Signed in as {who}**")
    acc1, acc2 = st.columns([1, 1])
    with acc1:
        theme_toggle("theme_app")
    with acc2:
        if st.button("Log out", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    st.markdown("---")
    st.markdown("## Enter relationship details")
    st.markdown("*Adjust sliders and dropdowns to describe the relationship.*")

    # ── Context
    st.markdown('<div class="sidebar-section">Context</div>', unsafe_allow_html=True)
    age = st.slider("Their age", 18, 55, 26, help="Age of the person being assessed")
    gender = st.selectbox("Their gender", ["Female", "Male", "Non-binary"])
    relationship_stage = st.selectbox(
        "How serious is the relationship?",
        ["Unrequited", "Non-established", "Casual dating",
         "Committed dating", "Cohabiting/Engaged", "Married"],
        index=2,
        help="Unrequited = one-sided interest · Non-established = just talking/early · "
             "then casual → committed → living together/engaged → married"
    )
    platform = st.selectbox(
        "Where do you mostly talk?",
        ["Dating app (Tinder/Bumble)", "Social media",
         "WhatsApp/Messaging", "In-person/Offline", "Mixed"],
        index=0,
        help="The main place your conversations happen"
    )
    relationship_duration_weeks = st.slider(
        "How long have you known each other? (weeks)", 1, 260, 8,
        help="Roughly how many weeks you've been in contact (260 weeks ≈ 5 years)"
    )

    # ── Communication patterns
    st.markdown('<div class="sidebar-section">Texting & talking habits</div>', unsafe_allow_html=True)
    message_frequency_per_day = st.slider(
        "How many messages do you exchange a day?", 0.2, 30.0, 6.0, step=0.1,
        help="Total texts back and forth on a typical day")
    avg_response_time_hours = st.slider(
        "How long do replies usually take? (hours)", 0.1, 72.0, 4.0, step=0.1,
        help="Average wait before they reply. 0.1 = almost instant · 24 = about a day · 72 = three days")
    max_silence_gap_days = st.slider(
        "Longest they've gone without messaging (days)", 0.0, 30.0, 1.5, step=0.1,
        help="The biggest recent gap with no contact at all")
    initiation_ratio = st.slider(
        "Who starts the conversations more?", 0.0, 1.0, 0.5, step=0.01,
        help="0 = they always start · 0.5 = evenly shared · 1 = you always start. "
             "Higher means you're doing most of the reaching out.")
    conv_length_trend = st.select_slider(
        "Are your chats getting longer or shorter lately?", options=[-2, -1, 0, 1, 2], value=0,
        format_func=lambda x: {-2:"Much shorter",-1:"A bit shorter",0:"About the same",1:"A bit longer",2:"Much longer"}[x],
        help="How the length of your conversations has changed recently")
    response_rate_pct = st.slider(
        "What share of your messages get a reply? (%)", 5.0, 100.0, 78.0, step=0.5,
        help="Out of every 100 messages you send, how many get answered")

    # ── Psychological
    st.markdown('<div class="sidebar-section">Feelings & personality</div>', unsafe_allow_html=True)
    rejection_sensitivity = st.slider(
        "How much do they worry about being rejected?", 1.0, 7.0, 3.5, step=0.1,
        help="1 = very secure and relaxed · 7 = very anxious about being turned down or abandoned")
    relationship_satisfaction = st.slider(
        "How happy are they in the relationship?", 1.0, 7.0, 4.5, step=0.1,
        help="1 = very unhappy · 7 = very happy and content")
    intimacy = st.slider(
        "How emotionally close do you feel?", 1.0, 7.0, 4.2, step=0.1,
        help="1 = distant, little sharing · 7 = very close, open and connected")
    passion = st.slider(
        "How strong is the romantic spark / attraction?", 1.0, 7.0, 4.3, step=0.1,
        help="1 = little excitement · 7 = strong attraction and excitement")
    commitment = st.slider(
        "How committed are they to making it last?", 1.0, 7.0, 3.8, step=0.1,
        help="1 = keeping it very casual · 7 = fully committed to the long term")
    perceived_social_support = st.slider(
        "How supported do they feel by friends & family?", 1.0, 7.0, 4.5, step=0.1,
        help="1 = feels alone with little support · 7 = strong support network around them")

    # ── Behavioural
    st.markdown('<div class="sidebar-section">Behaviour & history</div>', unsafe_allow_html=True)
    prior_ghosting_experience  = st.checkbox("They have been ghosted before",  value=False,
        help="Has this person been ghosted by someone in the past?")
    prior_ghosting_perpetrator = st.checkbox("They have ghosted someone before", value=False,
        help="Has this person ghosted someone else in the past?")
    breadcrumbing_exposure     = st.slider(
        "How often do they send mixed / non-committal signals?", 0.0, 10.0, 2.5, step=0.1,
        help="\"Breadcrumbing\" = occasional flirty texts or likes that keep you interested "
             "but never lead to real plans or commitment. 0 = never · 10 = constantly")
    neuroticism                = st.slider(
        "How easily do they get stressed or anxious (in general)?", 1.0, 5.0, 2.8, step=0.1,
        help="Their general temperament, not just in this relationship. "
             "1 = very calm and steady · 5 = easily worried, moody or reactive")
    active_matches             = st.slider(
        "How many other people are they chatting to?", 0, 20, 3,
        help="Number of other romantic conversations or matches going on at the same time")
    platform_inactivity        = st.checkbox(
        "They've recently gone quiet or inactive", value=False,
        help="Have they suddenly become much less active or stopped logging in?")
    conflict_frequency         = st.select_slider(
        "How often do you argue or clash?", options=[0, 1, 2, 3], value=1,
        format_func=lambda x: {0:"Never", 1:"Rarely", 2:"Sometimes", 3:"Often"}[x],
        help="How frequently disagreements or arguments come up")

    st.markdown("---")
    predict_btn = st.button("🔍  Predict ghosting risk", use_container_width=True, type="primary")

# ─── BUILD INPUT ROW ──────────────────────────────────────────────────────────

input_data = pd.DataFrame([{
    "age":                          age,
    "gender":                       gender,
    "relationship_stage":           relationship_stage,
    "platform":                     platform,
    "relationship_duration_weeks":  relationship_duration_weeks,
    "message_frequency_per_day":    message_frequency_per_day,
    "avg_response_time_hours":      avg_response_time_hours,
    "max_silence_gap_days":         max_silence_gap_days,
    "initiation_ratio":             initiation_ratio,
    "conv_length_trend":            conv_length_trend,
    "response_rate_pct":            response_rate_pct,
    "rejection_sensitivity":        rejection_sensitivity,
    "relationship_satisfaction":    relationship_satisfaction,
    "intimacy":                     intimacy,
    "passion":                      passion,
    "commitment":                   commitment,
    "perceived_social_support":     perceived_social_support,
    "prior_ghosting_experience":    int(prior_ghosting_experience),
    "prior_ghosting_perpetrator":   int(prior_ghosting_perpetrator),
    "breadcrumbing_exposure":       breadcrumbing_exposure,
    "neuroticism":                  neuroticism,
    "active_matches":               active_matches,
    "platform_inactivity":          int(platform_inactivity),
    "conflict_frequency":           conflict_frequency,
}])

# Friendly display names shared across charts
LABEL_MAP = {
    "response_rate_pct":            "Response rate (%)",
    "conv_length_trend":            "Conv. length trend",
    "prior_ghosting_perpetrator":   "Ghosted others before",
    "prior_ghosting_experience":    "Been ghosted before",
    "avg_response_time_hours":      "Avg response time (hrs)",
    "initiation_ratio":             "Initiation ratio",
    "rejection_sensitivity":        "Rejection sensitivity",
    "max_silence_gap_days":         "Max silence gap (days)",
    "platform_inactivity":          "Platform inactivity",
    "relationship_stage":           "Relationship stage",
    "relationship_satisfaction":    "Relationship satisfaction",
    "commitment":                   "Commitment",
    "breadcrumbing_exposure":       "Breadcrumbing exposure",
    "message_frequency_per_day":    "Messages per day",
    "relationship_duration_weeks":  "Relationship duration",
    "neuroticism":                  "Neuroticism",
    "conflict_frequency":           "Conflict frequency",
    "perceived_social_support":     "Social support",
    "active_matches":               "Active matches",
    "intimacy":                     "Intimacy",
    "passion":                      "Passion",
    "gender":                       "Gender",
    "platform":                     "Platform",
    "age":                          "Age",
}
def nice(name):
    return LABEL_MAP.get(name, name.replace("_", " ").capitalize())


def compute_contributions(input_row):
    """Deterministic per-prediction feature contributions for the linear model.

    For a linear model f(x) = w·x + b explained against a background sample, the
    exact (interventional) contribution of transformed feature i is

        phi_i = w_i · (x_i − mean_background_i)

    which is what SHAP's LinearExplainer computes — but here in closed form, so
    it needs no `shap` dependency and can never collapse to all-zeros the way a
    self-referential background did. One-hot dummies are summed back into their
    parent categorical feature for readability.

    Returns a DataFrame with columns [feature, contribution], or None if the
    model is not linear.
    """
    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "coef_"):
        return None

    x_t     = np.asarray(pre.transform(input_row))[0]
    bg_mean = np.asarray(pre.transform(load_background())).mean(axis=0)
    coef    = np.asarray(clf.coef_).ravel()
    phi     = coef * (x_t - bg_mean)                       # log-odds contribution

    nom_names = list(pre.named_transformers_["nom"].get_feature_names_out(nom_features))
    all_names = list(num_features) + list(ord_features) + nom_names

    agg = {}
    for name, val in zip(all_names, phi):
        base = next((f for f in nom_features if name.startswith(f + "_")), name)
        agg[base] = agg.get(base, 0.0) + float(val)

    df = pd.DataFrame({"feature": list(agg.keys()), "contribution": list(agg.values())})
    return df

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="gw-header">
    <h1>👻 GhostWatch</h1>
    <p>A machine learning tool for estimating the risk of sudden communication cessation (ghosting)
    in romantic relationships — for research and awareness purposes only.</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN PANEL ───────────────────────────────────────────────────────────────

if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">How it works</div>
            Adjust the relationship details in the left panel, then click
            <strong>Predict ghosting risk</strong>. The model analyses 24
            communication, psychological, and behavioural features to estimate risk.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">What it measures</div>
            The model predicts the likelihood of <em>sudden communication
            cessation</em> — unexplained withdrawal from a romantic
            relationship — based on patterns identified in relationship research.
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">Model performance</div>
            Logistic Regression trained on 2,000 observations.
            5-fold cross-validated AUC = <strong>0.953</strong> &nbsp;|&nbsp;
            F1 = <strong>0.817</strong> &nbsp;|&nbsp; Recall = <strong>0.885</strong>.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Feature importance (SHAP — mean |value| across training set)")
    top_n = importance_df.head(12).sort_values("importance", ascending=True)
    display_names = [nice(f) for f in top_n["feature"]]

    fig = go.Figure(go.Bar(
        x=top_n["importance"].values,
        y=display_names,
        orientation="h",
        marker_color=[TH["accent1"] if i >= len(top_n) - 3 else
                      TH["accent2"] if i >= len(top_n) - 6 else TH["bar_neutral"]
                      for i in range(len(top_n))],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    style_fig(fig, 420)
    fig.update_layout(xaxis_title="Mean |SHAP value|")
    st.plotly_chart(fig, use_container_width=True)

else:
    prob     = pipeline.predict_proba(input_data)[0][1]
    pred     = int(pipeline.predict(input_data)[0])
    risk_pct = round(prob * 100, 1)

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        if pred == 1:
            st.markdown(f"""
            <div class="risk-high">
                <p class="risk-label">⚠ Elevated Risk</p>
                <p class="risk-prob">Estimated ghosting probability: <strong>{risk_pct}%</strong></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <p class="risk-label">✓ Lower Risk</p>
                <p class="risk-prob">Estimated ghosting probability: <strong>{risk_pct}%</strong></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        gauge_color = TH["risk_high_text"] if pred == 1 else TH["risk_low_text"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 32, "family": "Lora, serif", "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TH["text_muted"], "tickfont": {"size": 10}},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": TH["gauge_bg"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35],   "color": TH["risk_low_bg"]},
                    {"range": [35, 60],  "color": TH["gauge_mid"]},
                    {"range": [60, 100], "color": TH["risk_high_bg"]},
                ],
                "threshold": {"line": {"color": TH["text"], "width": 2}, "thickness": 0.75, "value": 50},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(l=10, r=10, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter, sans-serif", "color": TH["text"]},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{risk_pct}%</div>
                <div class="metric-lbl">Risk score</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            verdict = "HIGH" if pred == 1 else "LOW"
            v_color = TH["risk_high_text"] if pred == 1 else TH["risk_low_text"]
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="color:{v_color}">{verdict}</div>
                <div class="metric-lbl">Risk level</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if pred == 1:
            notes = []
            if response_rate_pct < 60:            notes.append("Low response rate suggests reduced engagement.")
            if conv_length_trend <= -1:           notes.append("Declining conversation length is a key warning signal.")
            if avg_response_time_hours > 24:      notes.append("Very long average response times observed.")
            if max_silence_gap_days > 7:          notes.append("Extended silence gaps detected.")
            if prior_ghosting_perpetrator:        notes.append("Prior history of ghosting others increases risk.")
            if platform_inactivity:               notes.append("Partner's platform inactivity is a notable signal.")
            notes_html = "".join(f"<li>{n}</li>" for n in notes[:4])
            st.markdown(f"""
            <div class="gw-card">
                <div class="gw-card-title">Contributing signals</div>
                <ul style="padding-left:1.2rem;margin:0;font-size:0.85rem;color:{TH['text']}">{notes_html}</ul>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="gw-card">
                <div class="gw-card-title">Positive indicators</div>
                <p style="font-size:0.85rem;color:{TH['text']};margin:0">
                The current communication patterns and relationship indicators
                suggest a lower likelihood of sudden withdrawal. Maintaining
                open communication and consistent engagement supports relationship health.
                </p>
            </div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown("### Feature contribution to this prediction")
        st.caption("Bars show how each feature pushed the prediction toward or away from 'ghosted'. "
                   "Red = increases risk · Blue = reduces risk.")

        contrib_df = None
        try:
            contrib_df = compute_contributions(input_data)
        except Exception as e:
            st.warning(f"Detailed contribution unavailable ({e}); showing global importance instead.")

        if contrib_df is not None and contrib_df["contribution"].abs().max() > 1e-9:
            top = (contrib_df.reindex(contrib_df["contribution"].abs()
                                      .sort_values(ascending=False).index)
                   .head(14)
                   .sort_values("contribution"))
            display = [nice(f) for f in top["feature"]]
            colors  = [TH["shap_pos"] if v > 0 else TH["shap_neg"] for v in top["contribution"]]

            fig_contrib = go.Figure(go.Bar(
                x=top["contribution"].values,
                y=display,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
                text=[f"{v:+.3f}" for v in top["contribution"].values],
                textposition="outside",
                textfont={"size": 10, "color": TH["text"]},
            ))
            fig_contrib.add_vline(x=0, line_width=1, line_color=TH["text_muted"])
            style_fig(fig_contrib, 460, right_margin=60)
            fig_contrib.update_layout(xaxis_title="Contribution to ghosting risk (log-odds)")
            st.plotly_chart(fig_contrib, use_container_width=True)
        else:
            # Fallback: global importance (only if the model is non-linear or
            # every contribution is genuinely zero)
            top_imp = importance_df.head(10).sort_values("importance", ascending=True)
            fig_fb = go.Figure(go.Bar(
                x=top_imp["importance"].values,
                y=[nice(f) for f in top_imp["feature"]],
                orientation="h",
                marker_color=TH["accent1"],
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            style_fig(fig_fb, 400)
            fig_fb.update_layout(xaxis_title="Mean |SHAP value| (global)")
            st.plotly_chart(fig_fb, use_container_width=True)

        # ── Input summary (themed custom table)
        st.markdown("### Input summary")
        rows = [
            ("Relationship stage", relationship_stage),
            ("Platform", platform),
            ("Duration (weeks)", relationship_duration_weeks),
            ("Messages/day", message_frequency_per_day),
            ("Avg response time (hrs)", avg_response_time_hours),
            ("Max silence gap (days)", max_silence_gap_days),
            ("Initiation ratio", initiation_ratio),
            ("Conv. length trend", conv_length_trend),
            ("Response rate (%)", response_rate_pct),
            ("Relationship satisfaction", relationship_satisfaction),
            ("Commitment", commitment),
            ("Rejection sensitivity", rejection_sensitivity),
            ("Has ghosted before", "Yes" if prior_ghosting_perpetrator else "No"),
            ("Platform inactivity", "Yes" if platform_inactivity else "No"),
        ]
        table_html = "<table class='gw-table'>" + "".join(
            f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>" for k, v in rows
        ) + "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div class="disclaimer">
    <strong>Research tool only.</strong> GhostWatch is a final year project demonstration
    built on a theoretically grounded synthetic dataset. Predictions are probabilistic
    estimates, not definitive assessments of any individual's behaviour or intentions.
    Communication patterns alone cannot fully explain personal circumstances.
    This tool should not be used to make judgements about real people.
    &nbsp;·&nbsp; Model: Logistic Regression (best of 4 classifiers)
    &nbsp;·&nbsp; Validation AUC: 0.953 &nbsp;·&nbsp; F1: 0.817
</div>
""", unsafe_allow_html=True)
