import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Data: average personality scores + grouped MMLU results
# Replace with your actual averages if needed
data = {
    "Model": [
        "Llama-3-8b", "Literary Classicist", "Inventive Technologist",
        "Patent Strategist", "Cultural Scholar", "Technical Communicator",
        "Business Advisor", "Health Advisor", "Scientific Scholar",
        "Scientific Mathematician", "Legal Analyst", "Biomedical Expert"
    ],
    # # Big Five (means)
    # "Openness": [3.14, 3.26, 3.10, 3.03, 3.35, 3.01, 3.19, 3.33, 3.54, 2.94, 3.19, 2.79],
    # "Conscientiousness": [3.09, 3.01, 3.00, 3.01, 3.05, 3.14, 3.10, 3.01, 3.26, 2.83, 2.98, 3.03],
    # "Extraversion": [3.19, 3.19, 2.98, 3.00, 3.33, 2.82, 3.35, 3.25, 3.22, 2.92, 3.08, 3.14],
    # "Agreeableness": [2.69, 3.06, 3.09, 2.74, 2.90, 3.04, 3.07, 3.01, 2.91, 3.02, 3.07, 3.21],
    # "Neuroticism": [3.10, 3.19, 3.08, 3.18, 3.36, 2.99, 3.05, 3.65, 3.24, 2.85, 2.79, 2.98],
    "Openness": [
        3.13, 3.12, 2.93, 3.13, 2.88, 3.00,
        2.84, 3.27, 3.14, 2.95, 2.77, 3.05
    ],
    "Conscientiousness": [
        3.25, 3.19, 3.13, 3.30, 2.93, 2.99,
        3.22, 3.46, 3.13, 2.99, 3.10, 3.06
    ],
    "Extraversion": [
        3.33, 3.16, 2.91, 3.02, 2.93, 2.91,
        3.10, 3.16, 3.06, 2.99, 2.96, 3.13
    ],
    "Agreeableness": [
        3.07, 3.03, 3.11, 3.01, 3.01, 2.88,
        3.08, 3.02, 2.98, 3.14, 2.86, 3.05
    ],
    "Neuroticism": [
        3.14, 2.75, 2.87, 2.84, 3.06, 2.82,
        2.99, 3.15, 2.82, 2.98, 2.85, 3.03
    ],
    # Grouped MMLU performance (your aggregated table)
    "STEM": [0.5554, 0.5132, 0.5351, 0.5395, 0.4878, 0.5437, 0.5366, 0.4829, 0.4883, 0.5273, 0.5252, 0.5319],
    "Social_Science": [0.7774, 0.7474, 0.7565, 0.7719, 0.7015, 0.7684, 0.7570, 0.7011, 0.7030, 0.7659, 0.7498, 0.7714],
    "Humanities": [0.7158, 0.6904, 0.7154, 0.7117, 0.6562, 0.7119, 0.7142, 0.6580, 0.6470, 0.6933, 0.6993, 0.7079],
    "Health_Medicine": [0.7059, 0.6563, 0.6856, 0.6952, 0.6261, 0.6930, 0.6893, 0.6203, 0.6073, 0.6796, 0.6766, 0.6864],
    "Other": [0.6615, 0.6198, 0.6420, 0.6455, 0.5990, 0.6469, 0.6429, 0.5978, 0.6134, 0.6346, 0.6290, 0.6568],
}

df = pd.DataFrame(data)

# Traits and task groups
traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
task_groups = ["STEM", "Social_Science", "Humanities", "Health_Medicine", "Other"]

# Calculate correlations
results = []
for trait in traits:
    for task in task_groups:
        pearson_r, _ = pearsonr(df[trait], df[task])
        spearman_r, _ = spearmanr(df[trait], df[task])
        results.append((trait, task, round(pearson_r, 2), round(spearman_r, 2)))

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=["Trait", "Task Group", "Pearson r", "Spearman ρ"])
print(results_df)

# Optional: pivot for heatmap-style view
pivot_df = results_df.pivot(index="Trait", columns="Task Group", values="Pearson r")
print("\nPivot (Pearson r):")
print(pivot_df)