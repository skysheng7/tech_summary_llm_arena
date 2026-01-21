"""
Utility functions for connecting to OpenAI API and processing PDF files.
"""

import os
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
