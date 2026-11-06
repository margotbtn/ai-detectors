"""Homoglyph attack: swap Latin letters for visually identical Cyrillic/Greek ones.

The text looks unchanged on screen, but every swapped character becomes a
different, rare code point, so subword tokenizers split it into more,
less likely tokens.
"""

import random
import unicodedata

# Latin -> look-alike, Cyrillic where one exists, else Greek (N, Z). Same character
# set as RAID's homoglyph attack (checked against its rows in experiment 4); RAID
# picks Cyrillic or Greek at random for capitals, we always take the first.
HOMOGLYPHS = {
    "a": "\u0430", "c": "\u0441", "e": "\u0435", "i": "\u0456", "o": "\u043e", "p": "\u0440",
    "x": "\u0445", "y": "\u0443",
    "A": "\u0410", "B": "\u0412", "C": "\u0421", "E": "\u0415", "H": "\u041d", "I": "\u0406",
    "K": "\u041a", "M": "\u041c", "N": "\u039d", "O": "\u041e", "P": "\u0420", "T": "\u0422",
    "X": "\u0425", "Z": "\u0396",
}
# Greek alternatives RAID also uses for capitals: the defense must undo them too.
_GREEK_CAPITALS = {
    "A": "\u0391", "B": "\u0392", "E": "\u0395", "H": "\u0397", "I": "\u0399", "K": "\u039a",
    "M": "\u039c", "O": "\u039f", "P": "\u03a1", "T": "\u03a4", "X": "\u03a7",
}
_REVERSE = {v: k for k, v in {**HOMOGLYPHS}.items()} | {v: k for k, v in _GREEK_CAPITALS.items()}


def homoglyph_attack(text: str, p: float, seed: int = 0) -> str:
    """Replace a fraction `p` of the eligible characters (keys of HOMOGLYPHS).

    p = 0 returns the input unchanged; p = 1 replaces every eligible character.
    Deterministic for a given (text, p, seed), and NESTED across p: the
    characters swapped at p=5% are also swapped at p=10%, so a sweep over p
    adds swaps to the same text instead of redrawing them.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    eligible = [i for i, ch in enumerate(text) if ch in HOMOGLYPHS]
    k = round(p * len(eligible))
    if k == 0:
        return text
    random.Random(seed).shuffle(eligible)
    chosen = set(eligible[:k])
    return "".join(HOMOGLYPHS[ch] if i in chosen else ch for i, ch in enumerate(text))


ATTACK_PS = [0.0, 0.05, 0.10, 0.15, 0.20, 1.0]


def normalize_defense(text: str) -> str:
    """Cheap defense: NFKC normalization plus mapping known confusables back."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(_REVERSE.get(ch, ch) for ch in text)
