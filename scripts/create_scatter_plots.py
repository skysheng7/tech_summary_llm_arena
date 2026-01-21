import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv("all_judgements_meta.csv")

# Define score columns
score_columns = [
    "completeness_and_relevance.score",
    "factual_accuracy.score",
    "hallucination.score",
    "prompt_following.score",
    "research_question.score",
    "terminology_explanation_and_coherence.score",
]

# Define readable labels matching the bar plot naming convention
readable_labels = {
    "completeness_and_relevance.score": "Completeness & Relevance",
    "factual_accuracy.score": "Factual Accuracy",
    "hallucination.score": "Hallucination",
    "prompt_following.score": "Prompt Following",
    "research_question.score": "Research Question",
    "terminology_explanation_and_coherence.score": "Terminology & Coherence",
}

# Define colors using Paired palette
colors = plt.cm.Paired(np.linspace(0, 1, len(score_columns)))
color_map = {col: colors[i] for i, col in enumerate(score_columns)}

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))

# Pivot data to compare gemini vs openai scores
# We need to match papers with the same: paper_id, judge_prompt, generator_model, summary_style
gemini_df = df[df["judge_model"] == "gemini"].copy()
openai_df = df[df["judge_model"] == "openai"].copy()

# Merge on the matching columns
merged_df = gemini_df.merge(
    openai_df,
    on=["paper_id", "judge_prompt", "generator_model", "summary_style"],
    suffixes=("_gemini", "_openai"),
)

# Plot data points for each metric
for score_col in score_columns:
    gemini_col = f"{score_col}_gemini"
    openai_col = f"{score_col}_openai"

    x_values = merged_df[gemini_col].values
    y_values = merged_df[openai_col].values

    # Add jitter to avoid overlapping points
    jitter_strength = 0.1
    x_jitter = x_values + np.random.normal(0, jitter_strength, size=len(x_values))
    y_jitter = y_values + np.random.normal(0, jitter_strength, size=len(y_values))

    ax.scatter(
        x_jitter,
        y_jitter,
        c=[color_map[score_col]],
        label=readable_labels[score_col],
        alpha=0.6,
        s=100,
        edgecolors="black",
        linewidths=0.7,
    )

# Add diagonal reference line (y=x)
ax.plot([0, 10], [0, 10], "k--", alpha=0.3, linewidth=2, label="Perfect Agreement")

# Customize plot
ax.set_xlabel("Gemini Score", fontsize=18, fontweight="bold")
ax.set_ylabel("OpenAI Score", fontsize=18, fontweight="bold")
ax.set_title(
    "Judge Agreement: Gemini vs OpenAI Scores",
    fontsize=20,
    fontweight="bold",
)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.set_xticks(range(0, 11))
ax.set_yticks(range(0, 11))
ax.tick_params(axis="both", which="major", labelsize=14)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(
    title="Evaluation Metrics",
    fontsize=12,
    title_fontsize=13,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
)
ax.set_aspect("equal")

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig("data_analysis/score_scatter_plots.png", dpi=300, bbox_inches="tight")
print("Scatter plot saved to: data_analysis/score_scatter_plots.png")
