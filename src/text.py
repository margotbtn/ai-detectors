"""Text normalization and truncation applied to EVERY text before scoring or display.

Dependency-free on purpose: the app imports it to treat pasted text exactly
like the corpus.

Why: in RAID the two classes are formatted differently, and the formatting
alone separates them (found 2026-11-05 via the shallow-cue baseline, where
newline rate gave AUROC 0.92 on news and 0.00 on abstracts):
  news       human = single "\\n" between paragraphs, AI = blank lines "\\n\\n"
  abstracts  human = arXiv hard-wrapped at ~80 chars, leading/double spaces;
             chatgpt/gpt4 = no line breaks; llama-chat = blank lines
Perplexity scorers see those whitespace tokens, so without this step part of
any result would be measuring layout, not writing.
"""


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace (newlines, tabs, double spaces) to one space."""
    return " ".join(text.split())


def truncate(texts, tokenizer, max_tokens: int) -> list[str]:
    """Cut each text after `max_tokens` tokens, at the token's character offset.

    Cutting at the original character offset (rather than decode(ids[:n]))
    keeps the text identical up to the cut, whatever the tokenizer does.
    """
    texts = list(texts)
    enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
    return [text if len(offsets) <= max_tokens else text[: offsets[max_tokens - 1][1]]
            for text, offsets in zip(texts, enc["offset_mapping"])]
