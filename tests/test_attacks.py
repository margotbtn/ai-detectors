import unicodedata

import pytest

from src.attacks import HOMOGLYPHS, homoglyph_attack, normalize_defense

TEXT = "The quick brown fox jumps over the lazy dog. Copy Paste EXAMPLE text, 2024."


def test_p0_is_identity():
    assert homoglyph_attack(TEXT, 0.0) == TEXT


def test_attack_changes_text_but_keeps_length():
    out = homoglyph_attack(TEXT, 0.5)
    assert out != TEXT
    assert len(out) == len(TEXT)  # one code point for one code point


def test_attack_is_visually_identical_mapping():
    # Every substitute is a Cyrillic or Greek letter of the same case as the Latin original.
    for latin, sub in HOMOGLYPHS.items():
        assert unicodedata.name(sub).split()[0] in ("CYRILLIC", "GREEK")
        assert sub.isupper() == latin.isupper()
    out = homoglyph_attack(TEXT, 1.0)
    for a, b in zip(TEXT, out):
        assert a == b or HOMOGLYPHS[a] == b


def test_p1_replaces_every_eligible_char():
    out = homoglyph_attack(TEXT, 1.0)
    assert not any(ch in HOMOGLYPHS for ch in out)


def test_fraction_and_nesting():
    eligible = sum(ch in HOMOGLYPHS for ch in TEXT)
    swapped = {}
    for p in (0.1, 0.2, 0.5):
        out = homoglyph_attack(TEXT, p, seed=3)
        swapped[p] = {i for i, (a, b) in enumerate(zip(TEXT, out)) if a != b}
        assert len(swapped[p]) == round(p * eligible)
    assert swapped[0.1] <= swapped[0.2] <= swapped[0.5]


def test_deterministic():
    assert homoglyph_attack(TEXT, 0.3, seed=1) == homoglyph_attack(TEXT, 0.3, seed=1)


def test_defense_restores_original():
    for p in (0.05, 0.2, 1.0):
        assert normalize_defense(homoglyph_attack(TEXT, p)) == TEXT


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_invalid_p(p):
    with pytest.raises(ValueError):
        homoglyph_attack(TEXT, p)


def test_defense_undoes_greek_capitals_used_by_raid():
    # RAID sometimes uses Greek instead of Cyrillic capitals (e.g. Tau for T).
    assert normalize_defense("\u03a4he \u0391pple") == "The Apple"
