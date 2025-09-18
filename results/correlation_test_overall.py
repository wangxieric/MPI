import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Example data: replace with your actual model-level results
# Each row = one model, columns = traits and performance
# data = {
#     "Model": [
#         "Llama-3-8b", "Literary Classicist", "Inventive Technologist",
#         "Patent Strategist", "Cultural Scholar", "Technical Communicator",
#         "Business Advisor", "Health Advisor", "Scientific Scholar",
#         "Scientific Mathematician", "Legal Analyst", "Biomedical Expert"
#     ],
#     "Openness": [3.14, 3.26, 3.10, 3.03, 3.35, 3.01, 3.19, 3.33, 3.54, 2.94, 3.19, 2.79],
#     "Conscientiousness": [3.09, 3.01, 3.00, 3.01, 3.05, 3.14, 3.10, 3.01, 3.26, 2.83, 2.98, 3.03],
#     "Extraversion": [3.19, 3.19, 2.98, 3.00, 3.33, 2.82, 3.35, 3.25, 3.22, 2.92, 3.08, 3.14],
#     "Agreeableness": [2.69, 3.06, 3.09, 2.74, 2.90, 3.04, 3.07, 3.01, 2.91, 3.02, 3.07, 3.21],
#     "Neuroticism": [3.10, 3.19, 3.08, 3.18, 3.36, 2.99, 3.05, 3.65, 3.24, 2.85, 2.79, 2.98],
#     "MMLU": [0.6628, 0.6242, 0.6457, 0.6515, 0.5937, 0.6521, 0.6469, 0.5911, 0.5917, 0.6390, 0.6350, 0.6485]
# }

data = {
    "Model": [
        "Llama-3-8b", "Literary Classicist", "Inventive Technologist",
        "Patent Strategist", "Cultural Scholar", "Technical Communicator",
        "Business Advisor", "Health Advisor", "Scientific Scholar",
        "Scientific Mathematician", "Legal Analyst", "Biomedical Expert"
    ],
    "Openness": [
        3.1327433628318584, 3.1204819277108435, 2.932926829268293,
        3.127450980392157, 2.8773584905660377, 3.0000000000000000,
        2.840909090909091, 3.2705882352941176, 3.1417910447761193,
        2.9529411764705884, 2.7746478873239435, 3.046511627906977
    ],
    "Conscientiousness": [
        3.2535211267605635, 3.189873417721519, 3.132275132275132,
        3.299145299145299, 2.9279279279279278, 2.9872611464968153,
        3.2195121951219514, 3.4600000000000000, 3.134228187919463,
        2.989795918367347, 3.104575163398693, 3.0618556701030926
    ],
    "Extraversion": [
        3.3257575757575757, 3.164383561643836, 2.911242603550296,
        3.019047619047619, 2.9320388349514563, 2.9083969465648853,
        3.096774193548387, 3.1616161616161618, 3.0616438356164384,
        2.9908256880733943, 2.955223880597015, 3.127906976744186
    ],
    "Agreeableness": [
        3.066666666666667, 3.028169014084507, 3.106918238993711,
        3.0088495575221237, 3.0098039215686274, 2.879432624113475,
        3.081081081081081, 3.0229885057471266, 2.9782608695652173,
        3.1354166666666665, 2.861111111111111, 3.0481927710843375
    ],
    "Neuroticism": [
        3.1415929203539825, 2.753623188405797, 2.87248322147651,
        2.84375, 3.0568181818181817, 2.8203125,
        2.987012987012987, 3.1547619047619047, 2.8188976377952755,
        2.975, 2.8512396694214877, 3.026666666666667
    ],
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