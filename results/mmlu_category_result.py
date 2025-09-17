import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Input data
data = {
    "Category": ["STEM", "Social Science", "Humanities", "Health/Medicine", "Other"],
    "Llama-3-8b": [0.5553961401, 0.7774232945, 0.7157806686, 0.7059384226, 0.6615275955],
    "Literary Classicist": [0.5132105263, 0.7474, 0.6904444444, 0.6563, 0.6197777778],
    "Inventive Technologist": [0.5351403438, 0.75653956, 0.7153825407, 0.6855547918, 0.6420444233],
    "Patent Strategist": [0.5395204188, 0.7718872151, 0.7117056232, 0.695161377, 0.6454860432],
    "Cultural Scholar": [0.487778062, 0.7015443294, 0.6562373988, 0.6260869738, 0.5989872702],
    "Technical Communicator": [0.5436623053, 0.7683523898, 0.7119439033, 0.6929588602, 0.6468558429],
    "Business Advisor": [0.5365943361, 0.757023525, 0.7141697311, 0.6893351338, 0.6428711142],
    "Health Advisor": [0.4828724559, 0.701092716, 0.6579758575, 0.6203116193, 0.5978021138],
    "Scientific Scholar": [0.4882544783, 0.7030231622, 0.646962125, 0.6073442067, 0.61344947],
    "Scientific Mathematician": [0.5272617152, 0.7658851183, 0.6933374799, 0.679601142, 0.6346174522],
    "Legal Analyst": [0.5251751263, 0.7498263145, 0.6992652678, 0.6765778091, 0.6289817983],
    "Biomedical Expert": [0.531863114, 0.7713535272, 0.7079366365, 0.6863921003, 0.6567617694],
}

# Create DataFrame
df = pd.DataFrame(data)
df.set_index("Category", inplace=True)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Accuracy'})
plt.ylabel("Category", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=10, fontweight="bold")
plt.yticks(fontsize=10, fontweight="bold")

plt.tight_layout()
# Save and show
plt.savefig("mmlu_categories_heatmap.pdf", dpi=300)
plt.show()