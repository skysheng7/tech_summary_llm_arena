# Technical Document Summarization with LLM

Authors (listed in alphabetical order): Eshed Gal, Jeenat Mehareen, Marvin, Mitali, Sky Sheng

## About

This project provides a streamlined workflow for summarizing technical documents (i.e., research papers) using LLM from various providers (e.g., OpenAI, Gemini, etc.). The tool supports batch processing of PDF files with customizable prompts and flexible index-based selection for large document collections.

## Disclaimer

This project utilized Generative AI (specifically Claude Sonnet 4.5 within Cursor IDE) to assist with code generation. The system prompt we used to instruct Cursor's AI agents has been shared in the [.cursor/skills/api-rules/SKILL.md](.cursor/skills/api-rules/SKILL.md) file. All generated code has been thoroughly reviewed and edited by the authors to ensure it meets the specific requirements and standards of this project.

## Repository Structure

<details>
<summary>Click to expand</summary>

```
tech_summary_llm_arena/
├── input_docs/              # Place your PDF files here for summarization
├── results/                 # Generated summary text files
├── scripts/                 # Main Python scripts
├── notebook/                # Jupyter notebooks for experimentation
├── .cursor/skills/          # System prompts for Cursor
├── environment.yml          # Conda environment specifications
├── conda-lock.yml          # Locked dependency versions for various OS including: Linux, macOS, Windows
├── requirements.txt         # Python package dependencies
├── LICENSE                  # Project license
└── README.md               # Usage instructions
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

1. Add your OpenAI API key to the `.env` file, you can do the same for your Gemini API key:

```
OPENAI_API_KEY=<your_openai_api_key_here>
```

1. Obtain an API key:
   - Visit [OpenAI Platform](https://platform.openai.com/)
   - Sign up or log in
   - Navigate to API Keys section
   - Create a new secret key

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
