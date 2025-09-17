import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Example data: replace with your actual model-level results
# Each row = one model, columns = traits and performance
data = {
    "Model": [
        "Llama-3-8b", "Literary Classicist", "Inventive Technologist",
        "Patent Strategist", "Cultural Scholar", "Technical Communicator",
        "Business Advisor", "Health Advisor", "Scientific Scholar",
        "Scientific Mathematician", "Legal Analyst", "Biomedical Expert"
    ],
    "Openness": [3.14, 3.26, 3.10, 3.03, 3.35, 3.01, 3.19, 3.33, 3.54, 2.94, 3.19, 2.79],
    "Conscientiousness": [3.09, 3.01, 3.00, 3.01, 3.05, 3.14, 3.10, 3.01, 3.26, 2.83, 2.98, 3.03],
    "Extraversion": [3.19, 3.19, 2.98, 3.00, 3.33, 2.82, 3.35, 3.25, 3.22, 2.92, 3.08, 3.14],
    "Agreeableness": [2.69, 3.06, 3.09, 2.74, 2.90, 3.04, 3.07, 3.01, 2.91, 3.02, 3.07, 3.21],
    "Neuroticism": [3.10, 3.19, 3.08, 3.18, 3.36, 2.99, 3.05, 3.65, 3.24, 2.85, 2.79, 2.98],
    "MMLU": [0.6628, 0.6242, 0.6457, 0.6515, 0.5937, 0.6521, 0.6469, 0.5911, 0.5917, 0.6390, 0.6350, 0.6485]
}

df = pd.DataFrame(data)

# Compute correlations
traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

results = []
for trait in traits:
    pearson_r, _ = pearsonr(df[trait], df["MMLU"])
    spearman_r, _ = spearmanr(df[trait], df["MMLU"])
    results.append((trait, round(pearson_r, 2), round(spearman_r, 2)))

# Convert results into a DataFrame for nice display
results_df = pd.DataFrame(results, columns=["Trait", "Pearson r", "Spearman ρ"])
print(results_df)