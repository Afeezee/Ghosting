# Ghosting Prediction Dataset — Codebook

**Project:** Predictive Modeling of Sudden Communication Cessation in Romantic Relationships  
**File:** `ghosting_prediction_dataset.csv`  
**Rows:** 2,000 | **Columns:** 25 (24 features + 1 target)  
**Class balance:** Ghosted = 28.6% | Not ghosted = 71.4%

---

## Theoretical basis

Features are derived from the student's own literature review:

| Feature group | Source |
|---|---|
| Communication patterns | Schokkenbroek et al. (2025) |
| Rejection sensitivity | Mishra & Allen (2023) |
| Intimacy / Passion / Commitment / Stage | Cassepp-Borges et al. (2023) |
| Neuroticism, online dating personality | Bonilla-Zorita et al. (2021) |
| Prior ghosting experience / perpetration | Navarro et al. (2021) |
| Breadcrumbing, perceived social support | Jaspal & Lopes (2025) |

---

## Column Definitions

### Demographic & Context

| Column | Type | Range / Values | Description |
|---|---|---|---|
| `age` | int | 18–55 | Respondent age in years |
| `gender` | str | Male / Female / Non-binary | Self-reported gender |
| `relationship_stage` | str | 6 categories | Stage of relationship (ordered: Unrequited → Married) |
| `platform` | str | 5 categories | Primary communication platform |
| `relationship_duration_weeks` | int | 1–260 | Duration of the relationship in weeks |

### Communication Pattern Features

| Column | Type | Range | Description |
|---|---|---|---|
| `message_frequency_per_day` | float | 0.2–30 | Average number of messages exchanged daily |
| `avg_response_time_hours` | float | 0.1–72 | Average time to reply to a message (hours) |
| `max_silence_gap_days` | float | 0–30 | Longest stretch without any communication (days) |
| `initiation_ratio` | float | 0.03–0.98 | Proportion of conversations initiated by the subject (0=never, 1=always) |
| `conv_length_trend` | int | -2 to +2 | Trend in conversation length (-2=rapidly declining, 0=stable, +2=growing) |
| `response_rate_pct` | float | 5–100 | Percentage of messages received that were replied to |

### Psychological Features

| Column | Type | Range | Description |
|---|---|---|---|
| `rejection_sensitivity` | float | 1–7 | Tendency to anxiously expect / react to rejection (Likert 1–7) |
| `relationship_satisfaction` | float | 1–7 | Overall satisfaction with the relationship (1=very dissatisfied, 7=very satisfied) |
| `intimacy` | float | 1–7 | Felt closeness and emotional connection (Sternberg, 1–7) |
| `passion` | float | 1–7 | Romantic attraction and excitement (1–7) |
| `commitment` | float | 1–7 | Intention to maintain the relationship long-term (1–7) |
| `perceived_social_support` | float | 1–7 | Belief that others provide emotional / practical support (1–7) |

### Behavioural Features

| Column | Type | Range | Description |
|---|---|---|---|
| `prior_ghosting_experience` | int | 0 / 1 | Has the subject been ghosted before? (1=yes) |
| `prior_ghosting_perpetrator` | int | 0 / 1 | Has the subject ghosted someone before? (1=yes) |
| `breadcrumbing_exposure` | float | 0–10 | Degree of exposure to breadcrumbing behaviour (0=none, 10=high) |
| `neuroticism` | float | 1–5 | Neuroticism personality trait score (1=low, 5=high) |
| `active_matches` | int | 0–20 | Number of simultaneous active conversations / matches |
| `platform_inactivity` | int | 0 / 1 | Has the subject recently gone inactive on the platform? (1=yes) |
| `conflict_frequency` | int | 0–3 | Frequency of conflicts (0=none, 1=rare, 2=occasional, 3=frequent) |

### Target Variable

| Column | Type | Values | Description |
|---|---|---|---|
| `ghosted` | int | 0 / 1 | **Target.** Whether ghosting (sudden communication cessation) occurred. 1=ghosted, 0=not ghosted |

---

## Notes for the student

1. **Relationship stage** should be ordinally encoded (0–5) before modelling, not one-hot encoded, since it has a natural order.
2. **Gender** and **platform** should be one-hot encoded.
3. **conv_length_trend** is already numeric (−2 to +2) and requires no encoding.
4. **Class imbalance** is 71:29. Apply `class_weight='balanced'` in sklearn classifiers, or use SMOTE from `imbalanced-learn` on the training fold only.
5. All Likert-scale features (1–7) should be **standardised** before distance-based models (KNN, SVM).
6. Do **not** include `ghosted` as a feature — it is the target variable.
