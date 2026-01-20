import json
import requests
from google import genai
from google.genai import types
import os

def send_request(prompt, max_tokens):
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[prompt],
            config={
                'max_output_tokens': max_tokens,
                'temperature': 1.0,
                'thinking_config': {
                    'include_thoughts': True, 
                    'thinking_budget': 0 
                }
            },
        )
        if response:
            print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
            print(f"Thinking tokens: {response.usage_metadata.thoughts_token_count}")
            print(f"Output tokens (Answer): {response.usage_metadata.candidates_token_count}")
            print(f"Total tokens used: {response.usage_metadata.total_token_count}")
            return response.text
    except Exception as e:
        print(e)
    return ''



PERTURBATION_PROMPTS = {
    "paraphrase": (
        "Paraphrase the following summary using different wording and sentence "
        "structure. Do NOT add, remove, or change any information.\n\nSummary:\n{summary}"
    ),
    "long": (
        "Rewrite the following summary to be longer by restating ideas and adding "
        "redundant explanations. Do NOT introduce new facts.\n\nSummary:\n{summary}"
    ),
    "bullets": (
        "Convert the following summary into bullet points. Preserve all information "
        "and do not add anything new.\n\nSummary:\n{summary}"
    ),
    "shuffle": (
        "Reorder the sentences in the following summary. "
        "Do NOT change, rephrase, add, or remove any sentence content. "
        "Only change the order of sentences.\n\nSummary:\n{summary}"
    ),
}

# folder of original summaries
input_folder = "results/results_gemini"

for perturb_name, prompt_template in PERTURBATION_PROMPTS.items():
    # folder for perturbed summaries
    output_folder = f"results_gemini_{perturb_name}"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, "r", encoding="utf-8") as f:
            base_summary = f.read()

        prompt = prompt_template.format(summary=base_summary)

        print(f"Generating {perturb_name} for {filename}...")

        perturbed_text = send_request(
            prompt,
            max_tokens=700,
        )

        if perturbed_text:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(perturbed_text)
        else:
            print(f"Failed for {filename}")