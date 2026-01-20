
from openai_utils import *


client = load_openai_client()

file_id = upload_file(client, "input_docs/sky_prompt_revision_bias.pdf")

summary = summarize_file(client, file_id, prompt="Please summarize this document.")

print(summary)