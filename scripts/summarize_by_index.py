"""
Script to summarize PDF files by index range using any AI provider.
"""

import click
from general_utils import summarize_pdfs_by_index


@click.command()
@click.option(
    "--provider",
    default="anthropic",
    type=click.Choice(["openai", "anthropic", "google"], case_sensitive=False),
    help="AI provider to use for summarization (default: anthropic)",
)
@click.option(
    "--folder", default="input_docs", help="Path to folder containing PDF files"
)
@click.option(
    "--start", default=0, type=int, help="Starting index (0-based) of files to process"
)
@click.option(
    "--end",
    default=None,
    type=int,
    help="Ending index (exclusive) of files to process. If not specified, processes to the end",
)
@click.option(
    "--prompt",
    default="Can you provide a summary of this article in 5 sentences?",
    help="Prompt to guide the summarization",
)
@click.option("--output", default="results/results_anthropic_short", help="Folder to save summary text files")
@click.option(
    "--model",
    default="claude-sonnet-4-5-20250929",
    help="Model to use. If not specified, uses provider's default model",
)
@click.option("--max-tokens", default=50000, help="Maximum tokens in response")
@click.option("--temperature", default=1.0, help="Sampling temperature")
def main(provider, folder, start, end, prompt, output, model, max_tokens, temperature):
    """Summarize PDF files in a folder by index range using specified AI provider."""
    summarize_pdfs_by_index(
        provider=provider,
        folder_path=folder,
        start_index=start,
        end_index=end,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        output_folder=output,
    )


if __name__ == "__main__":
    main()
