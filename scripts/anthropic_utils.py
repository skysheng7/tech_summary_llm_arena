"""
Utility functions for connecting to Anthropic API and processing PDF files.
"""

import os
from dotenv import load_dotenv
import anthropic


def load_anthropic_client() -> anthropic.Anthropic:
    """
    Load Anthropic client with API key from environment variables.

    Returns
    -------
    anthropic.Anthropic
        Initialized Anthropic client

    Raises
    ------
    ValueError
        If ANTHROPIC_API_KEY is not found in environment variables
    """
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables. "
            "Please create a .env file with your API key."
        )

    client = anthropic.Anthropic(api_key=api_key)
    return client


def upload_file_anthropic(client: anthropic.Anthropic, file_path: str) -> str:
    """
    Upload a file to Anthropic for processing.

    Parameters
    ----------
    client : anthropic.Anthropic
        Initialized Anthropic client
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

    # Get file name
    file_name = os.path.basename(file_path)

    # Upload file to Anthropic
    file = client.beta.files.upload(
        file=(file_name, open(file_path, "rb"), "application/pdf")
    )

    return file.id


def summarize_file_anthropic(
    client: anthropic.Anthropic,
    file_id: str,
    prompt: str = "Please summarize this document.",
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 50000,
    temperature: float = 1.0,
) -> str:
    """
    Summarize a file using Anthropic API.

    Parameters
    ----------
    client : anthropic.Anthropic
        Initialized Anthropic client
    file_id : str
        File ID from Anthropic file upload
    prompt : str, optional
        Prompt to guide the summarization (default: "Please summarize this document.")
    model : str, optional
        Model to use for summarization (default: "claude-sonnet-4-5-20250929")
    max_tokens : int, optional
        Maximum tokens in response (default: 50000)
    temperature : float, optional
        Sampling temperature (default: 1.0)

    Returns
    -------
    str
        Generated summary text
    """
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
                        "text": prompt,
                    },
                ],
            }
        ],
        betas=["files-api-2025-04-14"],
    )

    return response.content[0].text
