"""
Utility functions for connecting to local OLLAMA API and processing PDF files.
"""

import os
import ollama
from PyPDF2 import PdfReader


def load_ollama_client() -> ollama.Client:
    """
    Load OLLAMA client for local model access.

    Returns
    -------
    ollama.Client
        Initialized OLLAMA client (points to localhost by default)

    Notes
    -----
    OLLAMA must be running locally. Start it with: ollama serve
    """
    client = ollama.Client()
    return client


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text content from a PDF file.

    Parameters
    ----------
    file_path : str
        Path to the PDF file

    Returns
    -------
    str
        Extracted text from all pages of the PDF

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text content from a txt file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Reading a txt file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    return text

def summarize_text_ollama(
    client: ollama.Client,
    text: str,
    prompt: str = "Can you provide a summary of this article in 5 sentences?",
    model: str = "llama3",
    temperature: float = 1.0,
) -> str:
    """
    Summarize text using OLLAMA API.

    Parameters
    ----------
    client : ollama.Client
        Initialized OLLAMA client
    text : str
        Text content to summarize
    prompt : str, optional
        Prompt to guide the summarization (default: "Please summarize this document.")
    model : str, optional
        Model to use for summarization (default: "llama3")
    temperature : float, optional
        Sampling temperature (default: 1.0)

    Returns
    -------
    str
        Generated summary text
    """
    full_prompt = f"{prompt}\n\n{text}"

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

    return response["message"]["content"]


def summarize_file_ollama(
    client: ollama.Client,
    file_path: str,
    prompt: str = "Can you provide a summary of this article in 5 sentences?",
    model: str = "llama3",
    temperature: float = 1.0,
) -> str:
    """
    Summarize a PDF file using OLLAMA API.

    This function extracts text from the PDF and sends it to OLLAMA for summarization.

    Parameters
    ----------
    client : ollama.Client
        Initialized OLLAMA client
    file_path : str
        Path to the PDF file
    prompt : str, optional
        Prompt to guide the summarization (default: "Please summarize this document.")
    model : str, optional
        Model to use for summarization (default: "llama3")
    temperature : float, optional
        Sampling temperature (default: 1.0)

    Returns
    -------
    str
        Generated summary text

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    """
    text = extract_text_from_pdf(file_path)
    summary = summarize_text_ollama(
        client=client,
        text=text,
        prompt=prompt,
        model=model,
        temperature=temperature,
    )
    return summary

