import re


def compile_phrase_regex(phrase: str):
    """Build a robust regex that matches a phrase, ignoring punctuation and casing.

    - Normalizes the phrase by stripping punctuation and lowercasing tokens
    - Ignores punctuation in the input by allowing non-alphanumerics between letters
    """
    raw_tokens = [t for t in re.split(r"\s+", phrase.strip()) if t]
    tokens = [re.sub(r"[^A-Za-z0-9]+", "", t.lower()) for t in raw_tokens]
    tokens = [t for t in tokens if t]
    if not tokens:
        return re.compile(r"(?!x)x")
    connector = r"[^A-Za-z0-9]*"

    def token_pattern(tok: str) -> str:
        chars = [re.escape(c) for c in tok]
        return connector.join(chars)

    parts = [token_pattern(tok) for tok in tokens]
    pattern = connector.join(parts)
    return re.compile(pattern, re.IGNORECASE)


