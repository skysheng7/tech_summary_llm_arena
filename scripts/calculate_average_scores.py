import pandas as pd
from pathlib import Path

# Define file paths
input_file = Path(__file__).parent.parent / "all_judgements_meta.csv"
output_dir = Path(__file__).parent.parent / "data_analysis"
output_file = output_dir / "average_scores.csv"

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# Read the input CSV
df = pd.read_csv(input_file)

# Define the columns to calculate averages for
score_columns = [
    "completeness_and_relevance.score",
    "factual_accuracy.score",
    "hallucination.score",
    "prompt_following.score",
    "research_question.score",
    "terminology_explanation_and_coherence.score",
    "total_score",
]

# Define grouping columns
group_columns = ["judge_model", "judge_prompt", "generator_model", "summary_style"]

# Calculate average scores for each group
average_scores = df.groupby(group_columns)[score_columns].mean().reset_index()

# Round to 2 decimal places for readability
average_scores[score_columns] = average_scores[score_columns].round(2)

# Sort by judge_model, judge_prompt, generator_model, and summary_style
average_scores = average_scores.sort_values(by=group_columns)

# Save to CSV
average_scores.to_csv(output_file, index=False)

print(f"Average scores calculated and saved to: {output_file}")
print(f"\nTotal number of unique groups: {len(average_scores)}")
print(f"\nFirst few rows:")
print(average_scores.head())
