"""
Utility functions for connecting to OpenAI API and processing PDF files.
"""

import os
import click
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv
from openai import OpenAI


def load_openai_client() -> OpenAI:
    """
    Load OpenAI client with API key from environment variables.

    Returns
    -------
    OpenAI
        Initialized OpenAI client

    Raises
    ------
    ValueError
        If OPENAI_API_KEY is not found in environment variables
    """
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please create a .env file with your API key."
        )

    client = OpenAI(api_key=api_key)
    return client


def upload_file_openai(client: OpenAI, file_path: str) -> str:
    """
    Upload a file to OpenAI for processing.

    Parameters
    ----------
    client : OpenAI
        Initialized OpenAI client
    file_path : str
        Path to the file to upload

    Returns
    -------
    str
        File ID of the uploaded file

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Upload file to OpenAI
    file = client.files.create(file=open(file_path, "rb"), purpose="user_data")

    return file.id


def summarize_file_openai(
    client: OpenAI,
    file_id: str,
    prompt: str = "Please summarize this document.",
    model: str = "gpt-5.2-2025-12-11",
    max_tokens: int = 50000,
    temperature: float = 1,
) -> str:
    """
    Summarize a file using OpenAI API.

    Parameters
    ----------
    client : OpenAI
        Initialized OpenAI client
    file_id : str
        File ID from OpenAI file upload
    prompt : str
        Prompt to guide the summarization
    model : str, optional
        Model to use for summarization (default: "gpt-5.2-2025-12-11")
    max_tokens : int, optional
        Maximum tokens in response (default: 50000)
    temperature : float, optional
        Sampling temperature (default: 1)

    Returns
    -------
    str
        Generated summary text
    """
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": file_id,
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    return response.output_text


def summarize_pdfs_in_folder_openai(
    folder_path: str = "input_docs",
    prompt: str = "Please summarize this document.",
    model: str = "gpt-5.2-2025-12-11",
    max_tokens: int = 50000,
    temperature: float = 1,
    output_folder: Optional[str] = "results",
) -> Dict[str, str]:
    """
    Iterate through all PDF files in a folder and summarize them.

    Parameters
    ----------
    folder_path : str, optional
        Path to folder containing PDF files (default: "input_docs")
    prompt : str, optional
        Prompt to guide the summarization (default: "Please summarize this document.")
    model : str, optional
        Model to use for summarization (default: "gpt-5.2-2025-12-11")
    max_tokens : int, optional
        Maximum tokens in response (default: 50000)
    temperature : float, optional
        Sampling temperature (default: 1)
    output_folder : str or None, optional
        Folder to save summary text files (default: "results").
        If None, summaries are not saved to disk.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping PDF filenames to their summaries

    Raises
    ------
    FileNotFoundError
        If the folder does not exist
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Initialize client
    client = load_openai_client()

    # Get all PDF files in folder
    pdf_files = list(Path(folder_path).glob("*.pdf"))

    if len(pdf_files) == 0:
        print(f"No PDF files found in {folder_path}")
        return {}

    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Process each PDF
    summaries = {}

    for i, pdf_path in enumerate(pdf_files):
        pdf_name = pdf_path.name

        try:
            # Upload file
            file_id = upload_file_openai(client, str(pdf_path))

            # Generate summary
            summary = summarize_file_openai(
                client=client,
                file_id=file_id,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Store summary
            summaries[pdf_name] = summary

            # Save to file if output folder specified
            if output_folder:
                output_path = os.path.join(
                    output_folder, pdf_name.replace(".pdf", "_summary.txt")
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(summary)

        except Exception as e:
            summaries[pdf_name] = f"Error: {e}"

    return summaries


def summarize_pdfs_by_index_openai(
    folder_path: str = "input_docs",
    start_index: int = 0,
    end_index: Optional[int] = None,
    prompt: str = "Please summarize this document.",
    model: str = "gpt-5.2-2025-12-11",
    max_tokens: int = 50000,
    temperature: float = 1,
    output_folder: Optional[str] = "results",
) -> Dict[str, str]:
    """
    Summarize PDF files in a folder by index range.

    Parameters
    ----------
    folder_path : str, optional
        Path to folder containing PDF files (default: "input_docs")
    start_index : int, optional
        Starting index (0-based) of files to process (default: 0)
    end_index : int or None, optional
        Ending index (exclusive) of files to process.
        If None, processes to the end (default: None)
    prompt : str, optional
        Prompt to guide the summarization (default: "Please summarize this document.")
    model : str, optional
        Model to use for summarization (default: "gpt-5.2-2025-12-11")
    max_tokens : int, optional
        Maximum tokens in response (default: 50000)
    temperature : float, optional
        Sampling temperature (default: 1)
    output_folder : str or None, optional
        Folder to save summary text files (default: "results").
        If None, summaries are not saved to disk.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping PDF filenames to their summaries

    Raises
    ------
    FileNotFoundError
        If the folder does not exist
    ValueError
        If indices are invalid
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Initialize client
    client = load_openai_client()

    # Get all PDF files in folder
    pdf_files = sorted(list(Path(folder_path).glob("*.pdf")))

    if len(pdf_files) == 0:
        print(f"No PDF files found in {folder_path}")
        return {}

    # Validate indices
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    if end_index is not None and end_index <= start_index:
        raise ValueError("end_index must be greater than start_index")

    if start_index >= len(pdf_files):
        raise ValueError(
            f"start_index {start_index} is out of range (total files: {len(pdf_files)})"
        )

    # Select files by index
    if end_index is None:
        selected_files = pdf_files[start_index:]
    else:
        selected_files = pdf_files[start_index:end_index]

    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Process selected PDFs
    summaries = {}

    for pdf_path in selected_files:
        pdf_name = pdf_path.name

        try:
            # Upload file
            file_id = upload_file_openai(client, str(pdf_path))

            # Generate summary
            summary = summarize_file_openai(
                client=client,
                file_id=file_id,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Store summary
            summaries[pdf_name] = summary

            # Save to file if output folder specified
            if output_folder:
                output_path = os.path.join(
                    output_folder, pdf_name.replace(".pdf", "_summary.txt")
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(summary)

        except Exception as e:
            summaries[pdf_name] = f"Error: {e}"

    return summaries


@click.command()
@click.option(
    "--folder", default="input_docs", help="Path to folder containing PDF files"
)
@click.option(
    "--prompt",
    default="Please summarize this document.",
    help="Prompt to guide the summarization",
)
@click.option("--output", default="results", help="Folder to save summary text files")
@click.option("--model", default="gpt-5.2-2025-12-11", help="OpenAI model to use")
@click.option("--max-tokens", default=50000, help="Maximum tokens in response")
@click.option("--temperature", default=1.0, help="Sampling temperature")
def main(folder, prompt, output, model, max_tokens, temperature):
    """Summarize all PDF files in a folder using OpenAI API."""
    summarize_pdfs_in_folder_openai(
        folder_path=folder,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        output_folder=output,
    )


if __name__ == "__main__":
    main()
