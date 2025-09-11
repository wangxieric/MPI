# --- Radar charts: one figure per variant (Baseline + Variant@120 + Variant@1k + Human) ---

import numpy as np
import matplotlib
matplotlib.use("Agg")   # safe backend
import matplotlib.pyplot as plt

# -----------------------------
# Constants and helpers
# -----------------------------
FACTORS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
K = len(FACTORS)
ANGLES = np.linspace(0, 2*np.pi, K, endpoint=False).tolist()
ANGLES += ANGLES[:1]  # close loop

def _close(vals):  # close polygon
    return vals + vals[:1]

def radar_one_variant(variant_key: str,
                      means_120: dict[str, list[float]],
                      means_1k: dict[str, list[float]],
                      human: list[float],
                      baseline_dataset: str = "1k",
                      ylim=(2.5, 4.2)):
    """Make one radar figure for a variant with dashed baseline & human."""
    if baseline_dataset not in {"1k", "120"}:
        raise ValueError("baseline_dataset must be '1k' or '120'.")

    # Baseline pick
    baseline = means_1k["Meta-Llama-3-8B"] if baseline_dataset == "1k" else means_120["Meta-Llama-3-8B"]

    # Collect the 4 series
    series = {
        f"Baseline (Meta-Llama-3-8B, {baseline_dataset})": (baseline, "dashed"),
        f"{variant_key} (120-item)": (means_120[variant_key], "solid"),
        f"{variant_key} (1k-item)":  (means_1k[variant_key], "solid"),
        "Human (120-item)": (human, "dashed"),
    }

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels(FACTORS)
    ax.set_ylim(*ylim)
    ax.set_title(f"Big Five — {variant_key} vs baseline & human")

    for label, (vals, style) in series.items():
        v = _close(vals)
        ax.plot(ANGLES, v, linewidth=2.0, label=label, linestyle=style)
        ax.fill(ANGLES, v, alpha=0.08) if style == "solid" else None  # fill only for solid lines

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=8, frameon=False)
    plt.tight_layout()
    safe_name = variant_key.replace("/", "_")
    plt.savefig(f"radar_{safe_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Data: means only (from your tables)
# -----------------------------

# 120-item MPI
means_120 = {
    "Meta-Llama-3-8B":          [3.67, 3.07, 2.78, 2.62, 3.92],
    "literary_classicist":      [3.29, 2.73, 3.24, 2.55, 3.72],
    "inventive_technologist":   [3.05, 2.96, 3.62, 3.33, 2.92],
    "patent_strategist":        [3.27, 3.44, 3.18, 2.87, 3.40],
    "cultural_scholar":         [3.14, 2.86, 3.82, 2.76, 3.30],
    "technical_communicator":   [3.16, 2.71, 2.65, 3.21, 2.86],
    "business_advisor":         [2.94, 3.11, 3.33, 2.67, 3.43],
    "health_advisor":           [2.73, 3.08, 3.71, 1.53, 3.37],
    "scientific_scholar":       [3.09, 2.86, 3.08, 2.70, 3.18],
    "scientific_mathematician": [2.94, 2.83, 3.50, 3.02, 4.15],
    "legal_analyst":            [2.93, 2.80, 2.54, 2.69, 2.94],
    "biomedical_expert":        [2.57, 3.26, 3.23, 3.24, 3.05],
}

# 1k-item MPI
means_1k = {
    "Meta-Llama-3-8B":          [3.14, 3.09, 3.19, 2.69, 3.10],
    "literary_classicist":      [3.26, 3.01, 3.19, 3.06, 3.19],
    "inventive_technologist":   [3.10, 3.00, 2.98, 3.09, 3.08],
    "patent_strategist":        [3.03, 3.01, 3.00, 2.74, 3.18],
    "cultural_scholar":         [3.35, 3.05, 3.33, 2.90, 3.36],
    "technical_communicator":   [3.01, 3.14, 2.82, 3.04, 2.99],
    "business_advisor":         [3.19, 3.10, 3.35, 3.07, 3.05],
    "health_advisor":           [3.33, 3.01, 3.25, 3.01, 3.65],
    "scientific_scholar":       [3.54, 3.26, 3.22, 2.91, 3.24],
    "scientific_mathematician": [2.94, 2.83, 2.92, 3.02, 2.85],
    "legal_analyst":            [3.19, 2.98, 3.08, 3.07, 2.79],
    "biomedical_expert":        [2.79, 3.03, 3.14, 3.21, 2.98],
}

# Human baseline (from the 120-item study)
human = [3.44, 3.60, 3.41, 3.66, 2.80]

# -----------------------------
# Make one figure per variant
# -----------------------------
variants = [
    "literary_classicist",
    "inventive_technologist",
    "patent_strategist",
    "cultural_scholar",
    "technical_communicator",
    "business_advisor",
    "health_advisor",
    "scientific_scholar",
    "scientific_mathematician",
    "legal_analyst",
    "biomedical_expert",
]

# Choose which dataset to use for the single baseline curve: "1k" or "120"
BASELINE_DATASET = "1k"

for v in variants:
    radar_one_variant(
        variant_key=v,
        means_120=means_120,
        means_1k=means_1k,
        human=human,
        baseline_dataset=BASELINE_DATASET,
        ylim=(2.5, 4.2)  # zoom; adjust to taste
    )

print("Saved one PNG per variant (radar_<variant>.pdf).")