"""
app.py
========
GhostWatch — Ghosting Risk Prediction Interface
Streamlit web application for inference using the trained ghosting_model.pkl

Run locally:
    streamlit run app.py

Deploy (free):
    1. Push app.py, ghosting_model.pkl, feature_importance.csv to a GitHub repo
    2. Go to share.streamlit.io → New app → connect the repo
    3. Set Main file: app.py  →  Deploy

Requirements (requirements.txt):
    streamlit>=1.35
    pandas>=2.0
    numpy>=1.24
    scikit-learn>=1.3
    xgboost>=2.0
    imbalanced-learn>=0.11
    shap>=0.44
    plotly>=5.18
    matplotlib>=3.8
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GhostWatch — Ghosting Risk Predictor",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ---- Typography & Base ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---- Page background ---- */
    .stApp { background-color: #F7F5F2; }
    section[data-testid="stSidebar"] { background-color: #EDE9E3; border-right: 1px solid #D6CFC5; }

    /* ---- Header ---- */
    .gw-header {
        background: linear-gradient(135deg, #3D2C2C 0%, #5C3D3D 100%);
        border-radius: 12px;
        padding: 2rem 2.4rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #F7F5F2;
    }
    .gw-header h1 {
        font-family: 'Lora', serif;
        font-size: 2rem;
        font-weight: 600;
        margin: 0 0 0.3rem;
        color: #F7F5F2;
    }
    .gw-header p {
        font-size: 0.92rem;
        opacity: 0.75;
        margin: 0;
        color: #F7F5F2;
    }

    /* ---- Cards ---- */
    .gw-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #E2DDD8;
        margin-bottom: 1rem;
    }
    .gw-card-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8C7B6E;
        margin-bottom: 0.6rem;
    }

    /* ---- Risk Badge ---- */
    .risk-high {
        background: #FDF0EE; border: 1.5px solid #D97060; border-radius: 8px;
        padding: 1.2rem 1.4rem; text-align: center;
    }
    .risk-low {
        background: #EFF5F0; border: 1.5px solid #5A9E72; border-radius: 8px;
        padding: 1.2rem 1.4rem; text-align: center;
    }
    .risk-label { font-family: 'Lora', serif; font-size: 1.5rem; font-weight: 600; margin: 0; }
    .risk-high .risk-label { color: #C04A38; }
    .risk-low  .risk-label { color: #3A7A52; }
    .risk-prob { font-size: 0.88rem; color: #6B5F57; margin-top: 0.3rem; }

    /* ---- Sidebar section labels ---- */
    .sidebar-section {
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.07em; color: #8C7B6E; margin: 1.2rem 0 0.4rem;
        border-bottom: 1px solid #C8BEB5; padding-bottom: 0.25rem;
    }

    /* ---- Disclaimer ---- */
    .disclaimer {
        font-size: 0.78rem; color: #9E9087; background: #F0ECE7;
        border-radius: 6px; padding: 0.8rem 1rem; border-left: 3px solid #C8BEB5;
    }

    /* ---- Metrics row ---- */
    .metric-box {
        background: #F7F5F2; border-radius: 8px; padding: 0.8rem 1rem;
        border: 1px solid #E2DDD8; text-align: center;
    }
    .metric-val { font-size: 1.5rem; font-weight: 600; color: #3D2C2C; }
    .metric-lbl { font-size: 0.75rem; color: #8C7B6E; margin-top: 0.1rem; }

    /* hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "ghosting_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_importance():
    imp_path = os.path.join(os.path.dirname(__file__), "feature_importance.csv")
    return pd.read_csv(imp_path)

try:
    artefact      = load_model()
    pipeline      = artefact["pipeline"]
    num_features  = artefact["num_features"]
    ord_features  = artefact["ord_features"]
    nom_features  = artefact["nom_features"]
    best_model    = artefact["best_model"]
    importance_df = load_importance()
    model_loaded  = True
except FileNotFoundError:
    model_loaded = False
    st.error("⚠️ Model file not found. Run `model_pipeline.py` first to generate `ghosting_model.pkl`.")
    st.stop()

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="gw-header">
    <h1>👻 GhostWatch</h1>
    <p>A machine learning tool for estimating the risk of sudden communication cessation (ghosting)
    in romantic relationships — for research and awareness purposes only.</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR — INPUT FORM ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Enter relationship details")
    st.markdown("*Adjust sliders and dropdowns to describe the relationship.*")

    # ── Demographic & Context
    st.markdown('<div class="sidebar-section">Context</div>', unsafe_allow_html=True)

    age = st.slider("Age", 18, 55, 26, help="Age of the person being assessed")
    gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
    relationship_stage = st.selectbox(
        "Relationship stage",
        ["Unrequited", "Non-established", "Casual dating",
         "Committed dating", "Cohabiting/Engaged", "Married"],
        index=2,
        help="How far along the relationship is"
    )
    platform = st.selectbox(
        "Primary communication platform",
        ["Dating app (Tinder/Bumble)", "Social media",
         "WhatsApp/Messaging", "In-person/Offline", "Mixed"],
        index=0
    )
    relationship_duration_weeks = st.slider(
        "Relationship duration (weeks)", 1, 260, 8,
        help="How long has the relationship been ongoing?"
    )

    # ── Communication Patterns
    st.markdown('<div class="sidebar-section">Communication patterns</div>', unsafe_allow_html=True)

    message_frequency_per_day = st.slider(
        "Messages per day", 0.2, 30.0, 6.0, step=0.1,
        help="Average number of messages exchanged daily"
    )
    avg_response_time_hours = st.slider(
        "Average response time (hours)", 0.1, 72.0, 4.0, step=0.1,
        help="Average time taken to reply to a message"
    )
    max_silence_gap_days = st.slider(
        "Longest silence gap (days)", 0.0, 30.0, 1.5, step=0.1,
        help="Longest stretch without any communication"
    )
    initiation_ratio = st.slider(
        "Conversation initiation ratio", 0.0, 1.0, 0.5, step=0.01,
        help="Proportion of conversations started by this person (0=never, 1=always)"
    )
    conv_length_trend = st.select_slider(
        "Conversation length trend",
        options=[-2, -1, 0, 1, 2],
        value=0,
        format_func=lambda x: {-2:"Rapidly declining",-1:"Declining",0:"Stable",1:"Growing",2:"Rapidly growing"}[x],
        help="Trend in how long conversations have been recently"
    )
    response_rate_pct = st.slider(
        "Response rate (%)", 5.0, 100.0, 78.0, step=0.5,
        help="Percentage of messages that receive a reply"
    )

    # ── Psychological
    st.markdown('<div class="sidebar-section">Psychological indicators</div>', unsafe_allow_html=True)

    rejection_sensitivity = st.slider(
        "Rejection sensitivity (1–7)", 1.0, 7.0, 3.5, step=0.1,
        help="Tendency to anxiously expect and react to rejection"
    )
    relationship_satisfaction = st.slider(
        "Relationship satisfaction (1–7)", 1.0, 7.0, 4.5, step=0.1,
        help="Overall satisfaction with the relationship"
    )
    intimacy = st.slider(
        "Intimacy (1–7)", 1.0, 7.0, 4.2, step=0.1,
        help="Felt closeness and emotional connection (Sternberg)"
    )
    passion = st.slider(
        "Passion (1–7)", 1.0, 7.0, 4.3, step=0.1,
        help="Romantic attraction and excitement (Sternberg)"
    )
    commitment = st.slider(
        "Commitment (1–7)", 1.0, 7.0, 3.8, step=0.1,
        help="Intention to maintain the relationship long-term (Sternberg)"
    )
    perceived_social_support = st.slider(
        "Perceived social support (1–7)", 1.0, 7.0, 4.5, step=0.1,
        help="Belief that others provide emotional/practical support"
    )

    # ── Behavioural
    st.markdown('<div class="sidebar-section">Behavioural factors</div>', unsafe_allow_html=True)

    prior_ghosting_experience   = st.checkbox("Has been ghosted before",  value=False)
    prior_ghosting_perpetrator  = st.checkbox("Has ghosted someone before", value=False)
    breadcrumbing_exposure      = st.slider(
        "Breadcrumbing exposure (0–10)", 0.0, 10.0, 2.5, step=0.1,
        help="Degree of exposure to inconsistent/non-committal behaviour from partner"
    )
    neuroticism                 = st.slider(
        "Neuroticism (1–5)", 1.0, 5.0, 2.8, step=0.1,
        help="Personality trait: emotional instability and anxiety"
    )
    active_matches              = st.slider(
        "Active simultaneous conversations/matches", 0, 20, 3,
        help="Number of other people being messaged at the same time"
    )
    platform_inactivity         = st.checkbox(
        "Partner recently went inactive on platform", value=False
    )
    conflict_frequency          = st.select_slider(
        "Conflict frequency",
        options=[0, 1, 2, 3],
        value=1,
        format_func=lambda x: {0:"None", 1:"Rare", 2:"Occasional", 3:"Frequent"}[x],
        help="How often conflicts or arguments occur"
    )

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

# ─── MAIN PANEL ───────────────────────────────────────────────────────────────

if not predict_btn:
    # Default landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">How it works</div>
            Adjust the relationship details in the left panel, then click
            <strong>Predict ghosting risk</strong>. The model analyses 24
            communication, psychological, and behavioural features to estimate risk.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">What it measures</div>
            The model predicts the likelihood of <em>sudden communication
            cessation</em> — unexplained withdrawal from a romantic
            relationship — based on patterns identified in relationship research.
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="gw-card">
            <div class="gw-card-title">Model performance</div>
            Logistic Regression trained on 2,000 observations.
            5-fold cross-validated AUC = <strong>0.953</strong> &nbsp;|&nbsp;
            F1 = <strong>0.817</strong> &nbsp;|&nbsp; Recall = <strong>0.885</strong>.
        </div>
        """, unsafe_allow_html=True)

    # Feature importance chart (always visible)
    st.markdown("---")
    st.markdown("### Feature importance (SHAP — mean |value| across training set)")
    top_n = importance_df.head(12).sort_values("importance", ascending=True)

    # Pretty display names
    label_map = {
        "response_rate_pct":            "Response rate",
        "conv_length_trend":            "Conversation length trend",
        "prior_ghosting_perpetrator":   "Has ghosted someone before",
        "avg_response_time_hours":      "Average response time",
        "initiation_ratio":             "Initiation ratio",
        "rejection_sensitivity":        "Rejection sensitivity",
        "max_silence_gap_days":         "Longest silence gap",
        "platform_inactivity":          "Platform inactivity",
        "relationship_stage":           "Relationship stage",
        "relationship_satisfaction":    "Relationship satisfaction",
        "commitment":                   "Commitment",
        "breadcrumbing_exposure":       "Breadcrumbing exposure",
    }
    display_names = [label_map.get(f, f.replace("_", " ").capitalize()) for f in top_n["feature"]]

    fig = go.Figure(go.Bar(
        x=top_n["importance"].values,
        y=display_names,
        orientation="h",
        marker_color=["#C04A38" if i >= len(top_n) - 3 else
                      "#D97060" if i >= len(top_n) - 6 else "#B5A89A"
                      for i in range(len(top_n))],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        height=420,
        margin=dict(l=10, r=30, t=10, b=30),
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#E2DDD8", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # ── PREDICTION ──────────────────────────────────────────────────────────

    prob      = pipeline.predict_proba(input_data)[0][1]
    pred      = int(pipeline.predict(input_data)[0])
    risk_pct  = round(prob * 100, 1)

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        # Risk badge
        if pred == 1:
            st.markdown(f"""
            <div class="risk-high">
                <p class="risk-label">⚠ Elevated Risk</p>
                <p class="risk-prob">Estimated ghosting probability: <strong>{risk_pct}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <p class="risk-label">✓ Lower Risk</p>
                <p class="risk-prob">Estimated ghosting probability: <strong>{risk_pct}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability gauge
        gauge_color = "#C04A38" if pred == 1 else "#3A7A52"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 32, "family": "Lora, serif",
                                             "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#8C7B6E", "tickfont": {"size": 10}},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "#F0ECE7",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35],  "color": "#EFF5F0"},
                    {"range": [35, 60], "color": "#FDF7EE"},
                    {"range": [60, 100],"color": "#FDF0EE"},
                ],
                "threshold": {
                    "line": {"color": "#3D2C2C", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig_gauge.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Key metrics
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{risk_pct}%</div>
                <div class="metric-lbl">Risk score</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            verdict = "HIGH" if pred == 1 else "LOW"
            v_color = "#C04A38" if pred == 1 else "#3A7A52"
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="color:{v_color}">{verdict}</div>
                <div class="metric-lbl">Risk level</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Contextual note
        if pred == 1:
            notes = []
            if response_rate_pct < 60:
                notes.append("Low response rate suggests reduced engagement.")
            if conv_length_trend <= -1:
                notes.append("Declining conversation length is a key warning signal.")
            if avg_response_time_hours > 24:
                notes.append("Very long average response times observed.")
            if max_silence_gap_days > 7:
                notes.append("Extended silence gaps detected.")
            if prior_ghosting_perpetrator:
                notes.append("Prior history of ghosting others increases risk.")
            if platform_inactivity:
                notes.append("Partner's platform inactivity is a notable signal.")
            notes_html = "".join(f"<li>{n}</li>" for n in notes[:4])
            st.markdown(f"""
            <div class="gw-card">
                <div class="gw-card-title">Contributing signals</div>
                <ul style="padding-left:1.2rem;margin:0;font-size:0.85rem;color:#5C3D3D">{notes_html}</ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="gw-card">
                <div class="gw-card-title">Positive indicators</div>
                <p style="font-size:0.85rem;color:#3A5C44;margin:0">
                The current communication patterns and relationship indicators
                suggest a lower likelihood of sudden withdrawal. Maintaining
                open communication and consistent engagement supports relationship health.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        st.markdown("### Feature contribution to this prediction")
        st.caption("Bars show how each feature pushed the prediction toward or away from 'ghosted'. "
                   "Red = increases risk · Blue = reduces risk.")

        # Approximate SHAP-style contribution from preprocessed input
        # (True per-instance SHAP computed here via LinearExplainer)
        try:
            import shap as shap_lib
            pre = pipeline.named_steps["preprocessor"]
            clf = pipeline.named_steps["classifier"]
            X_transformed = pre.transform(input_data)

            # Get feature names after transformation
            num_names = num_features
            ord_names = ord_features
            nom_names = list(pre.named_transformers_["nom"].get_feature_names_out(nom_features))
            all_names = num_names + ord_names + nom_names

            explainer   = shap_lib.LinearExplainer(clf, shap_lib.maskers.Independent(
                              X_transformed, max_samples=100))
            shap_vals   = explainer(X_transformed)
            sv          = shap_vals.values[0]

            contrib_df = pd.DataFrame({"feature": all_names, "shap": sv})
            contrib_df["abs"] = contrib_df["shap"].abs()
            contrib_df = contrib_df.sort_values("abs", ascending=False).head(14)
            contrib_df = contrib_df.sort_values("shap", ascending=True)

            # Nice labels
            label_map = {
                "response_rate_pct":            "Response rate (%)",
                "conv_length_trend":            "Conv. length trend",
                "prior_ghosting_perpetrator":   "Ghosted others before",
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
            }
            display = [label_map.get(f, f.replace("_", " ").capitalize())
                       for f in contrib_df["feature"]]
            colors  = ["#C04A38" if v > 0 else "#4A7FA0" for v in contrib_df["shap"]]

            fig_contrib = go.Figure(go.Bar(
                x=contrib_df["shap"].values,
                y=display,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                text=[f"{v:+.3f}" for v in contrib_df["shap"].values],
                textposition="outside",
                textfont={"size": 10},
            ))
            fig_contrib.add_vline(x=0, line_width=1, line_color="#8C7B6E")
            fig_contrib.update_layout(
                xaxis_title="SHAP value (contribution to ghosting risk)",
                yaxis_title="",
                height=460,
                margin=dict(l=10, r=60, t=10, b=30),
                font=dict(family="Inter, sans-serif", size=12),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#E2DDD8", zeroline=False),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_contrib, use_container_width=True)

        except Exception as e:
            st.info(f"SHAP contribution chart unavailable: {e}")
            # Fallback: show top feature importances from file
            top_imp = importance_df.head(10).sort_values("importance", ascending=True)
            st.bar_chart(top_imp.set_index("feature")["importance"])

        # ── Input summary table
        st.markdown("### Input summary")
        summary = pd.DataFrame({
            "Feature": [
                "Relationship stage", "Platform", "Duration (weeks)",
                "Messages/day", "Avg response time (hrs)", "Max silence gap (days)",
                "Initiation ratio", "Conv. length trend", "Response rate (%)",
                "Relationship satisfaction", "Commitment", "Rejection sensitivity",
                "Has ghosted before", "Platform inactivity",
            ],
            "Value": [
                relationship_stage, platform, relationship_duration_weeks,
                message_frequency_per_day, avg_response_time_hours, max_silence_gap_days,
                initiation_ratio, conv_length_trend, response_rate_pct,
                relationship_satisfaction, commitment, rejection_sensitivity,
                "Yes" if prior_ghosting_perpetrator else "No",
                "Yes" if platform_inactivity else "No",
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True,
                     height=int(35 * len(summary) + 38))

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
