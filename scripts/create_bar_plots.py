import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv("data_analysis/average_scores.csv")

# Filter for judge_prompt == 'full' and summary_style == 'short'
filtered_df = df[(df["judge_prompt"] == "full") & (df["summary_style"] == "short")]

# Define score columns and their readable labels
score_columns = [
    "completeness_and_relevance.score",
    "factual_accuracy.score",
    "hallucination.score",
    "prompt_following.score",
    "research_question.score",
    "terminology_explanation_and_coherence.score",
    "total_score",
]

readable_labels = [
    "Completeness &\nRelevance",
    "Factual\nAccuracy",
    "Hallucination",
    "Prompt\nFollowing",
    "Research\nQuestion",
    "Terminology &\nCoherence",
    "Total Score\n(÷10)",
]

# Get unique generator models and judge models
generator_models = sorted(filtered_df["generator_model"].unique())
judge_models = sorted(filtered_df["judge_model"].unique())

# Define colors for judge models
colors = plt.cm.Set3(np.linspace(0, 1, len(judge_models)))
color_map = {judge: colors[i] for i, judge in enumerate(judge_models)}

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Create bar plots for each generator model
for idx, generator in enumerate(generator_models):
    ax = axes[idx]

    # Get data for this generator model
    generator_data = filtered_df[filtered_df["generator_model"] == generator]

    # Set up x positions
    x = np.arange(len(score_columns))
    width = 0.8 / len(judge_models)

    # Plot bars for each judge model
    for i, judge in enumerate(judge_models):
        judge_data = generator_data[generator_data["judge_model"] == judge]

        if not judge_data.empty:
            # Get scores and scale total_score by dividing by 10
            scores = []
            for col in score_columns:
                value = judge_data[col].values[0]
                if col == "total_score":
                    value = value / 10
                scores.append(value)

            # Plot bars
            offset = (i - len(judge_models) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                scores,
                width,
                label=judge.capitalize(),
                color=color_map[judge],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

    # Customize subplot
    ax.set_xlabel("Evaluation Metrics", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title(f"Generator: {generator.capitalize()}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(readable_labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Judge Model", fontsize=9)

# Overall title
fig.suptitle(
    "Model Performance Comparison\n(Full Judge Prompt, Short Summary Style)",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
plt.savefig(
    "data_analysis/model_comparison_bar_plots.png", dpi=300, bbox_inches="tight"
)
print("Bar plot saved to: data_analysis/model_comparison_bar_plots.png")
