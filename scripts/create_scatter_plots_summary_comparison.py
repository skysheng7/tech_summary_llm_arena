import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv("all_judgements_meta.csv")

# Filter for only gemini judge
gemini_df = df[df["judge_model"] == "gemini"].copy()

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

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()

# Define the summary styles to compare against 'short'
comparison_styles = ["long", "bullets", "shuffle", "paraphrase"]

# Create each subplot
for idx, comparison_style in enumerate(comparison_styles):
    ax = axes[idx]

    # Get data for short and comparison style
    short_df = gemini_df[gemini_df["summary_style"] == "short"].copy()
    comparison_df = gemini_df[gemini_df["summary_style"] == comparison_style].copy()

    # Merge on the matching columns
    merged_df = short_df.merge(
        comparison_df,
        on=["paper_id", "judge_prompt", "generator_model"],
        suffixes=("_short", f"_{comparison_style}"),
    )

    # Plot data points for each metric
    for score_col in score_columns:
        short_col = f"{score_col}_short"
        comparison_col = f"{score_col}_{comparison_style}"

        x_values = merged_df[short_col].values
        y_values = merged_df[comparison_col].values

        # Add jitter to avoid overlapping points
        jitter_strength = 0.1
        x_jitter = x_values + np.random.normal(0, jitter_strength, size=len(x_values))
        y_jitter = y_values + np.random.normal(0, jitter_strength, size=len(y_values))

        # Only add label for the first subplot (for the shared legend)
        label = readable_labels[score_col] if idx == 0 else None

        ax.scatter(
            x_jitter,
            y_jitter,
            c=[color_map[score_col]],
            label=label,
            alpha=0.6,
            s=80,
            edgecolors="black",
            linewidths=0.7,
        )

    # Add diagonal reference line (y=x)
    ax.plot([0, 10], [0, 10], "k--", alpha=0.3, linewidth=2)

    # Customize subplot
    ax.set_xlabel("Short Summary Score", fontsize=16, fontweight="bold")
    ax.set_ylabel(
        f"{comparison_style.capitalize()} Summary Score",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_title(
        f"Short vs {comparison_style.capitalize()}", fontsize=18, fontweight="bold"
    )
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 11))
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_aspect("equal")

# Create a single legend for the entire figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Evaluation Metrics",
    fontsize=12,
    title_fontsize=13,
    loc="center left",
    bbox_to_anchor=(1.01, 0.75),
)

# Overall title
fig.suptitle(
    "Gemini Judge: Summary Style Comparison (vs Short)",
    fontsize=22,
    fontweight="bold",
    y=0.995,
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
plt.savefig(
    "data_analysis/summary_style_comparison_scatter_plots.png",
    dpi=300,
    bbox_inches="tight",
)
print("Scatter plot saved to: data_analysis/summary_style_comparison_scatter_plots.png")
