"""
Judge summaries using Anthropic with PDF file attachment support.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict
from anthropic_utils import load_anthropic_client, upload_file_anthropic
from general_utils import extract_text_from_txt


def judge_single_summary(
    client,
    judge_prompt_path: str,
    paper_pdf_path: str,
    summary_text_path: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> Dict:
    """
    Judge a single summary using Anthropic with PDF file attachment.

    Parameters
    ----------
    client : anthropic.Anthropic
        Initialized Anthropic client
    judge_prompt_path : str
        Path to the judge prompt file (e.g., llm_judge_prompts/judge_basic.txt)
    paper_pdf_path : str
        Path to the original PDF paper
    summary_text_path : str
        Path to the summary text file
    model : str, optional
        Model to use for judging (default: "claude-sonnet-4-5-20250929")
    max_tokens : int, optional
        Maximum tokens in response (default: 4096)
    temperature : float, optional
        Sampling temperature (default: 0.2)

    Returns
    -------
    Dict
        JSON output from the judge model
    """
    judge_prompt_template = extract_text_from_txt(judge_prompt_path)
    summary_text = extract_text_from_txt(summary_text_path)

    file_id = upload_file_anthropic(client, paper_pdf_path)

    full_prompt = judge_prompt_template.replace(
        "{file_id}", f"[PDF file attached]"
    ).replace("{summary}", f"{{{summary_text}}}")

    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": file_id,
                        },
                    },
                    {
                        "type": "text",
                        "text": full_prompt,
                    },
                ],
            }
        ],
        betas=["files-api-2025-04-14"],
    )

    response_text = response.content[0].text

    try:
        response_text = response_text.strip()

        json_start = response_text.find("{")
        if json_start != -1:
            json_end = response_text.rfind("}")
            if json_end != -1:
                response_text = response_text[json_start : json_end + 1]

        result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        return {
            "_save_as_text": True,
            "raw_response": response_text,
        }


def judge_all_summaries(
    judge_prompt_path: str,
    summary_folder: str,
    input_docs_folder: str = "input_docs",
    output_folder: Optional[str] = None,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> Dict[str, Dict]:
    """
    Judge all summaries in a summary folder using the specified judge prompt.

    Parameters
    ----------
    judge_prompt_path : str
        Path to the judge prompt file (e.g., "llm_judge_prompts/judge_basic.txt")
    summary_folder : str
        Path to folder containing summary text files (e.g., "results/results_anthropic_short")
    input_docs_folder : str, optional
        Path to folder containing original PDF papers (default: "input_docs")
    output_folder : str or None, optional
        Folder to save judge result JSON files. If None, creates a folder in the format
        anthropic_judge_results_{judgeType}/{summary_folder_name} (default: None)
    model : str, optional
        Model to use for judging (default: "claude-sonnet-4-5-20250929")
    max_tokens : int, optional
        Maximum tokens in response (default: 4096)
    temperature : float, optional
        Sampling temperature (default: 0.2)

    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping PDF filenames to their judge results
    """
    if not os.path.exists(input_docs_folder):
        raise FileNotFoundError(f"Input docs folder not found: {input_docs_folder}")

    if not os.path.exists(summary_folder):
        raise FileNotFoundError(f"Summary folder not found: {summary_folder}")

    if not os.path.exists(judge_prompt_path):
        raise FileNotFoundError(f"Judge prompt not found: {judge_prompt_path}")

    if output_folder is None:
        summary_folder_name = Path(summary_folder).name
        judge_filename = Path(judge_prompt_path).stem
        judge_type = judge_filename.split("_")[-1]
        output_folder = os.path.join(
            f"anthropic_judge_results_{judge_type}", summary_folder_name
        )

    os.makedirs(output_folder, exist_ok=True)

    client = load_anthropic_client()

    summary_files = list(Path(summary_folder).glob("*.txt"))

    print(f"Found {len(summary_files)} summary files")

    if len(summary_files) == 0:
        print("No .txt files found, skipping...")
        return {}

    results = {}

    for summary_path in summary_files:
        summary_filename = summary_path.name
        print(f"\nProcessing: {summary_filename}")

        parts = summary_filename.rsplit("_", 1)
        original_pdf_name = parts[0] + ".pdf"

        # Check if already processed
        output_json = os.path.join(output_folder, parts[0] + "_judge.json")
        output_txt = os.path.join(output_folder, parts[0] + "_judge.txt")
        if os.path.exists(output_json) or os.path.exists(output_txt):
            print(f"  ⏭️  Already processed, skipping...")
            continue

        original_pdf_path = os.path.join(input_docs_folder, original_pdf_name)

        if not os.path.exists(original_pdf_path):
            print(f"  ❌ PDF not found: {original_pdf_name}")
            results[original_pdf_name] = {
                "error": f"Original PDF not found: {original_pdf_path}"
            }
            continue

        print(f"  ✓ Found PDF: {original_pdf_name}")

        try:
            judge_result = judge_single_summary(
                client=client,
                judge_prompt_path=judge_prompt_path,
                paper_pdf_path=original_pdf_path,
                summary_text_path=str(summary_path),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            results[original_pdf_name] = judge_result

            if judge_result.get("_save_as_text", False):
                parts = summary_filename.rsplit("_", 1)
                output_filename = parts[0] + "_judge.txt"
                output_path = os.path.join(output_folder, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(judge_result["raw_response"])
                print(f"  ✓ Saved as text: {output_filename}")
            else:
                parts = summary_filename.rsplit("_", 1)
                output_filename = parts[0] + "_judge.json"
                output_path = os.path.join(output_folder, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(judge_result, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Saved as JSON: {output_filename}")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[original_pdf_name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Iterate through all folders in summary_short
    summary_short_path = Path("summary_short")

    for folder in summary_short_path.iterdir():
        if folder.is_dir():
            summary_folder = str(folder)
            print(f"\n{'='*60}")
            print(f"Processing folder: {summary_folder}")
            print(f"{'='*60}")

            results = judge_all_summaries(
                judge_prompt_path="llm_judge_prompts/judge_full.txt",
                summary_folder=summary_folder,
                input_docs_folder="input_docs",
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=0.2,
            )

            print(f"\nCompleted {summary_folder}: {len(results)} results")
