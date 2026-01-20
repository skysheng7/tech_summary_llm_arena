"""
Judge summaries using llama3 with predefined judge prompts.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict
import ollama
from ollama_utils import (
    extract_text_from_pdf,
    load_ollama_client,
)
from general_utils import extract_text_from_txt


def judge_single_summary(
    client: ollama.Client,
    judge_prompt_path: str,
    paper_pdf_path: str,
    summary_text_path: str,
    model: str = "llama3",
    temperature: float = 0.2,
) -> Dict:
    """
    Judge a single summary using llama3.

    Parameters
    ----------
    client : ollama.Client
        Initialized OLLAMA client
    judge_prompt_path : str
        Path to the judge prompt file (e.g., llm_judge_prompts/judge_basic.txt)
    paper_pdf_path : str
        Path to the original PDF paper
    summary_text_path : str
        Path to the summary text file
    model : str, optional
        Model to use for judging (default: "llama3")
    temperature : float, optional
        Sampling temperature (default: 0.5)

    Returns
    -------
    Dict
        JSON output from the judge model
    """
    judge_prompt_template = extract_text_from_txt(judge_prompt_path)
    paper_text = extract_text_from_pdf(paper_pdf_path)
    summary_text = extract_text_from_txt(summary_text_path)

    full_prompt = judge_prompt_template.replace(
        "{file_id}", f"{{{paper_text}}}"
    ).replace("{summary}", f"{{{summary_text}}}")

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": full_prompt,
            }
        ],
        options={
            "temperature": temperature,
        },
    )

    response_text = response["message"]["content"]

    try:
        response_text = response_text.strip()

        # Find the first '{' which marks the start of JSON
        json_start = response_text.find("{")
        if json_start != -1:
            # Find the last '}' which marks the end of JSON
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
    model: str = "llama3",
    temperature: float = 0.5,
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
        {model}_judge_results_{judgeType}/{summary_folder_name} (default: None)
    model : str, optional
        Model to use for judging (default: "llama3")
    temperature : float, optional
        Sampling temperature (default: 0.5)

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
        summary_model = summary_folder_name.split("_")[1]
        output_folder = os.path.join(
            f"{summary_model}_judge_results_{judge_type}", summary_folder_name
        )

    os.makedirs(output_folder, exist_ok=True)

    client = load_ollama_client()

    summary_files = list(Path(summary_folder).glob("*_summary.txt"))

    if len(summary_files) == 0:
        return {}

    results = {}

    for summary_path in summary_files:
        summary_filename = summary_path.name

        original_pdf_name = summary_filename.replace("_summary.txt", ".pdf")

        original_pdf_path = os.path.join(input_docs_folder, original_pdf_name)

        if not os.path.exists(original_pdf_path):
            results[original_pdf_name] = {
                "error": f"Original PDF not found: {original_pdf_path}"
            }
            continue

        try:
            judge_result = judge_single_summary(
                client=client,
                judge_prompt_path=judge_prompt_path,
                paper_pdf_path=original_pdf_path,
                summary_text_path=str(summary_path),
                model=model,
                temperature=temperature,
            )

            results[original_pdf_name] = judge_result

            if judge_result.get("_save_as_text", False):
                output_filename = summary_filename.replace("_summary.txt", "_judge.txt")
                output_path = os.path.join(output_folder, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(judge_result["raw_response"])
            else:
                output_filename = summary_filename.replace(
                    "_summary.txt", "_judge.json"
                )
                output_path = os.path.join(output_folder, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(judge_result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            results[original_pdf_name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Example usage: Judge all summaries in results_anthropic_short using judge_basic.txt
    results_basic = judge_all_summaries(
        judge_prompt_path="llm_judge_prompts/judge_basic.txt",
        summary_folder="results/results_anthropic_short",
        input_docs_folder="input_docs",
        model="llama3",
        temperature=0.2,
    )
"""
    # Example usage: Judge all summaries in results_anthropic_short using judge_full.txt
    results_full = judge_all_summaries(
        judge_prompt_path="llm_judge_prompts/judge_full.txt",
        summary_folder="results/results_anthropic_short",
        input_docs_folder="input_docs",
        model="llama3",
        temperature=0.5,
    )
"""
