# Technical Document Summarization with LLM

Authors (listed in alphabetical order): Eshed Gal, Jeenat Mehareen, Marvin, Mitali, Sky (Kehan) Sheng

## About

This project provides a streamlined workflow for summarizing technical documents (i.e., research papers) using LLM from various providers (e.g., OpenAI, Gemini, etc.). The tool supports batch processing of PDF files with customizable prompts and flexible index-based selection for large document collections.

## Disclaimer

This project utilized Generative AI (specifically Claude Sonnet 4.5 within Cursor IDE) to assist with code generation. The system prompt we used to instruct Cursor's AI agents has been shared in the [.cursor/skills/api-rules/SKILL.md](.cursor/skills/api-rules/SKILL.md) file. All generated code has been thoroughly reviewed and edited by the authors to ensure it meets the specific requirements and standards of this project.

## Repository Structure

<details>
<summary>Click to expand</summary>

```
tech_summary_llm_arena/
├── .cursor/
│   └── skills/
│       └── api-rules/
│           └── SKILL.md     # System prompts for Cursor AI
├── input_docs/              # Place your PDF files here for summarization
├── results/                 # Generated outputs organized by analysis stage
│   ├── 00_not_used_results/ # Archived/unused results
│   ├── 01_summarize_docs/   # Initial document summaries
│   │   ├── results_anthropic_short/
│   │   ├── results_gemini_short/
│   │   ├── results_llama3_short/
│   │   └── results_openai_short/
│   ├── 02_summary_perturbations/ # Perturbed summary variations
│   │   ├── pertubations_anthropic_summaries/
│   │   │   ├── results_anthropic_bullets/
│   │   │   ├── results_anthropic_long/
│   │   │   ├── results_anthropic_paraphrase/
│   │   │   └── results_anthropic_shuffle/
│   │   ├── pertubations_llama3_summaries/
│   │   │   ├── results_llama3_bullets/
│   │   │   ├── results_llama3_long/
│   │   │   ├── results_llama3_paraphrase/
│   │   │   └── results_llama3_shuffle/
│   │   ├── perturbations_gemini_summaries/
│   │   │   ├── results_gemini_bullets/
│   │   │   ├── results_gemini_long/
│   │   │   ├── results_gemini_paraphrase/
│   │   │   └── results_gemini_shuffle/
│   │   └── perturbatios_openai_summaries/
│   │       ├── results_openai_bullets/
│   │       ├── results_openai_long/
│   │       ├── results_openai_paraphrase/
│   │       └── results_openai_shuffle/
│   ├── 03_llm_judges/       # LLM judge evaluation results
│   │   ├── anthropic_judge_results_full/
│   │   ├── gemini_judge_results_basic/
│   │   ├── gemini_judge_results_full/
│   │   ├── llama3_judge_results_basic/
│   │   ├── llama3_judge_results_full/
│   │   ├── llm_judge_prompts/
│   │   ├── openai_judge_results_basic/
│   │   └── openai_judge_results_full/
│   └── 04_data_analysis/    # Analysis visualizations and data
│       ├── all_judgements_meta.csv
│       ├── average_scores.csv
│       ├── model_comparison_bar_plots.png
│       ├── prompt_comparison_scatter_plots.png
│       ├── score_scatter_plots.png
│       └── summary_style_comparison_scatter_plots.png
├── scripts/                 # Python scripts for summarization and analysis
├── notebook/
├── .gitignore               # Git ignore rules
├── environment.yml          # Conda environment specifications
├── conda-lock.yml           # Locked dependency versions for various OS
├── LICENSE                  # Project license
└── README.md                # Usage instructions
```

</details>

## Dependencies

<details>
<summary>Click to expand</summary>

### Environment Management

- Conda (recommended)
- conda-lock (for reproducible environments)
- OLLAMA (for local model inference)

</details>

## Installation

<details>
<summary>Click to expand</summary>

1. Clone this repository:

```bash
git clone https://github.com/skysheng7/tech_summary_llm_arena.git
cd tech_summary_llm_arena
```

### Option 1: Using `environment.yml` (Quick Setup)

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate llm_judge
```

### Option 2: Using Conda-Lock (Reproducible Setup)

For exact reproducibility across different operating systems:

```bash
conda-lock install --name llm_judge conda-lock.yml
conda activate llm_judge
```

</details>

## Setup

<details>
<summary>Click to expand</summary>

### API Key Configuration

1. Create a `.env` file in the project root directory using command line:

```bash
touch .env
```

1. Add your API keys to the `.env` file:

```
OPENAI_API_KEY=<your_openai_api_key_here>
ANTHROPIC_API_KEY=<your_anthropic_api_key_here>
GEMINI_API_KEY=<your_gemini_api_key_here>
```

1. Obtain API keys:
   - **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/), sign up/log in, navigate to API Keys section, and create a new secret key
   - **Anthropic**: Visit [Anthropic Console](https://console.anthropic.com/), sign up/log in, and create an API key
   - **Gemini**: Visit [Google AI Studio](https://aistudio.google.com/), sign in with your Google account, and generate an API key

⚠️ **Important**: The `.env` file contains sensitive information and is automatically excluded from version control via `.gitignore`.

### Prepare Your Documents

Place the PDF files you want to summarize in the `input_docs/` folder.

</details>

## Usage

<details>
<summary>Click to expand</summary>

### Basic Usage

Summarize all PDF files in the `input_docs/` folder:

```bash
python scripts/openai_utils.py
```

### Summarize by Index Range

Process a specific range of files using index-based selection:

```bash
python scripts/summarize_by_index.py --start=0 --end=5
```

This will process the first 5 PDF files (indices 0-4).

### Advanced Options

Customize the summarization with additional parameters:

```bash
python scripts/summarize_by_index.py \
    --folder=input_docs \
    --start=0 \
    --end=10 \
    --prompt="Provide a detailed technical summary of this document, focusing on methodology and key findings." \
    --output=results \
    --model=gpt-5.2-2025-12-11 \
    --max-tokens=50000 \
    --temperature=1.0
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--folder` | `input_docs` | Path to folder containing PDF files |
| `--start` | `0` | Starting index (0-based) of files to process |
| `--end` | `None` | Ending index (exclusive); if not specified, processes to end |
| `--prompt` | `"Please summarize this document."` | Custom prompt for summarization |
| `--output` | `results` | Folder to save summary text files |
| `--model` | `gpt-5.2-2025-12-11` | OpenAI model to use |
| `--max-tokens` | `50000` | Maximum tokens in response |
| `--temperature` | `1.0` | Sampling temperature (0.0-2.0) |

### Output Format

Summary files are saved as plain text files in the `results/` folder with the naming convention:

```
[original_pdf_name]_summary.txt
```

For example, if your input file is `research_paper.pdf`, the summary will be saved as `research_paper_summary.txt`.

</details>

## Cost Considerations

<details>
<summary>Click to expand</summary>

⚠️ **API Usage Costs**: This tool makes calls to OpenAI's API, which incurs charges based on:

- Number of tokens processed (input PDF content + output summary)
- Model used (different models have different pricing)
- Number of files processed

**Recommendations:**

1. Test with a small batch first (e.g., `--start=0 --end=2`)
2. Monitor your API usage in the [OpenAI dashboard](https://platform.openai.com/usage)
3. Set usage limits in your OpenAI account settings
4. Use index-based processing to control batch sizes

Refer to [OpenAI's pricing page](https://openai.com/pricing) for current rates.

</details>
