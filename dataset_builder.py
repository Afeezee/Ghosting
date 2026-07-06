"""
Adjust ghosting rate — literature suggests 25-35% ghosting prevalence in dating.
Recalibrate threshold so we get ~28% positive class, which is realistic and
gives the model enough positive examples to learn from.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

age = np.random.normal(26, 5, N).clip(18, 55).astype(int)
gender = np.random.choice(['Male', 'Female', 'Non-binary'], N, p=[0.45, 0.50, 0.05])
relationship_stage = np.random.choice(
    ['Unrequited', 'Non-established', 'Casual dating', 'Committed dating', 'Cohabiting/Engaged', 'Married'],
    N, p=[0.08, 0.15, 0.28, 0.27, 0.14, 0.08]
)
stage_map = {'Unrequited': 0, 'Non-established': 1, 'Casual dating': 2,
             'Committed dating': 3, 'Cohabiting/Engaged': 4, 'Married': 5}
stage_ord = np.array([stage_map[s] for s in relationship_stage])
platform = np.random.choice(
    ['Dating app (Tinder/Bumble)', 'Social media', 'WhatsApp/Messaging', 'In-person/Offline', 'Mixed'],
    N, p=[0.30, 0.20, 0.18, 0.17, 0.15]
)
relationship_duration_weeks = np.where(
    stage_ord <= 1, np.random.randint(1, 6, N),
    np.where(stage_ord <= 2, np.random.randint(2, 24, N),
    np.where(stage_ord <= 3, np.random.randint(8, 104, N),
             np.random.randint(26, 260, N))))

baseline_freq = np.random.normal(5 + stage_ord * 2, 3, N).clip(0.5, 30)
message_frequency_per_day = np.random.normal(baseline_freq, 1.5, N).clip(0.2, 30).round(1)
avg_response_time_hours = np.random.exponential(scale=6, size=N).clip(0.1, 72).round(1)
max_silence_gap_days = np.random.exponential(scale=2 + (5 - stage_ord) * 0.5, size=N).clip(0, 30).round(1)
initiation_ratio = np.random.beta(3, 3, N).round(2)
conv_length_trend = np.random.choice([-2, -1, 0, 1, 2], N, p=[0.10, 0.18, 0.40, 0.20, 0.12])
response_rate_pct = np.random.normal(75, 18, N).clip(5, 100).round(1)
rejection_sensitivity = np.random.normal(3.5, 1.2, N).clip(1, 7).round(1)
relationship_satisfaction = np.random.normal(4.2, 1.3, N).clip(1, 7).round(1)
intimacy = np.random.normal(4.0 + stage_ord * 0.3, 1.2, N).clip(1, 7).round(1)
passion = np.random.normal(4.5 - np.abs(stage_ord - 2) * 0.3, 1.2, N).clip(1, 7).round(1)
commitment = np.random.normal(2.5 + stage_ord * 0.7, 1.1, N).clip(1, 7).round(1)
perceived_social_support = np.random.normal(4.5, 1.2, N).clip(1, 7).round(1)
prior_ghosting_experience = np.random.choice([0, 1], N, p=[0.55, 0.45])
prior_ghosting_perpetrator = np.random.choice([0, 1], N, p=[0.65, 0.35])
breadcrumbing_exposure = (np.random.beta(2, 5, N) * 10).round(1)
neuroticism = np.random.normal(2.8, 0.9, N).clip(1, 5).round(1)
active_matches = np.random.poisson(3, N).clip(0, 20)
platform_inactivity = np.random.choice([0, 1], N, p=[0.70, 0.30])
conflict_frequency = np.random.choice([0, 1, 2, 3], N, p=[0.30, 0.35, 0.25, 0.10])

risk_score = (
    0.20 * (1 - response_rate_pct / 100) +
    0.15 * np.log1p(avg_response_time_hours) / np.log1p(72) +
    0.15 * (max_silence_gap_days / 30) +
    0.10 * (1 - initiation_ratio) +
    0.10 * (-conv_length_trend + 2) / 4 +
    0.08 * (rejection_sensitivity / 7) +
    0.07 * (1 - relationship_satisfaction / 7) +
    0.05 * (1 - commitment / 7) +
    0.04 * prior_ghosting_perpetrator +
    0.03 * (neuroticism / 5) +
    0.03 * (breadcrumbing_exposure / 10) +
    0.03 * platform_inactivity +
    0.02 * (active_matches / 20) +
    0.05 * (1 - stage_ord / 5)
)
risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

# Use percentile-based threshold to get ~28% ghosting
threshold = np.percentile(risk_score, 72)  # top 28% = ghosted
noise = np.random.normal(0, 0.05, N)
ghosted = ((risk_score + noise) > threshold).astype(int)

print(f"Class distribution — Ghosted: {ghosted.sum()} ({ghosted.mean()*100:.1f}%) | Not ghosted: {(1-ghosted).sum()} ({(1-ghosted).mean()*100:.1f}%)")

df = pd.DataFrame({
    'age': age, 'gender': gender, 'relationship_stage': relationship_stage,
    'platform': platform, 'relationship_duration_weeks': relationship_duration_weeks,
    'message_frequency_per_day': message_frequency_per_day,
    'avg_response_time_hours': avg_response_time_hours,
    'max_silence_gap_days': max_silence_gap_days,
    'initiation_ratio': initiation_ratio, 'conv_length_trend': conv_length_trend,
    'response_rate_pct': response_rate_pct, 'rejection_sensitivity': rejection_sensitivity,
    'relationship_satisfaction': relationship_satisfaction, 'intimacy': intimacy,
    'passion': passion, 'commitment': commitment,
    'perceived_social_support': perceived_social_support,
    'prior_ghosting_experience': prior_ghosting_experience,
    'prior_ghosting_perpetrator': prior_ghosting_perpetrator,
    'breadcrumbing_exposure': breadcrumbing_exposure,
    'neuroticism': neuroticism, 'active_matches': active_matches,
    'platform_inactivity': platform_inactivity, 'conflict_frequency': conflict_frequency,
    'ghosted': ghosted
})

out_path = '/mnt/user-data/outputs/ghosting_prediction_dataset.csv'
df.to_csv(out_path, index=False)

# Quick audit
print(f"\nSaved to: {out_path}")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nFeature types:\n{df.dtypes.to_string()}")
