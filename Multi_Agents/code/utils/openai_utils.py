import json
import re
import tiktoken


model2max_context = {
    "gemini-3.1-flash-lite": 1000000,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
}

class OutOfQuotaException(Exception):
    """Raised when the API account has no remaining credits."""

    def __init__(self):
        super().__init__(
            "The Gemini API quota is exhausted. Check the quota and billing "
            "for your Google AI Studio project, or try again after its limit resets."
        )

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def parse_model_json(response: str) -> dict:
    """Parse a JSON object returned by a model, including fenced JSON output."""
    cleaned = response.strip()
    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match:
        cleaned = fenced_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if object_match:
            cleaned = object_match.group(0)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response did not contain valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed
