import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create the dataframe
# data = {
#     "Trait": ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"],
#     "Health/Medicine": [-0.03, -0.29, -0.50, -0.68, -0.79],
#     "Humanities": [0.08, -0.25, -0.47, -0.64, -0.75],
#     "STEM": [-0.10, -0.15, -0.51, -0.68, -0.71],
#     "Social Science": [0.01, -0.31, -0.56, -0.72, -0.83],
#     "Other": [-0.04, -0.04, -0.44, -0.66, -0.75]
# }

data = {
    "Trait": ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"],
    "Health/Medicine": [0.14, -0.07, 0.06, -0.05, -0.29],
    "Humanities": [0.14, -0.01, 0.04, -0.13, -0.29],
    "Other": [0.17, -0.08, 0.19, -0.04, -0.13],
    "STEM": [0.12, -0.07, 0.10, -0.10, -0.24],
    "Social Science": [0.17, -0.10, 0.11, -0.15, -0.18]
}

df = pd.DataFrame(data)
df.set_index("Trait", inplace=True)

# Clustered heatmap
g = sns.clustermap(
    df, annot=True, cmap="coolwarm", center=0, fmt=".2f",
    figsize=(8, 6), cbar_kws={'label': 'Correlation'}
)

# rotate x tick labels for better readability
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

# Change font properties of x and y tick labels
for label in g.ax_heatmap.get_xticklabels():
    label.set_fontsize(12)           # change font size
    label.set_fontname("Arial")      # change font family
    label.set_fontweight("bold")     # change font weight

for label in g.ax_heatmap.get_yticklabels():
    label.set_fontsize(12)
    label.set_fontname("Arial")
    label.set_fontweight("bold")

# plt.suptitle("Clustered Correlation Heatmap of Traits and Task Groups", y=1.02, fontsize=14)
plt.savefig("correlation.pdf", dpi=300, bbox_inches="tight")
plt.show()