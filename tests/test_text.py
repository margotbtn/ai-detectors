from src.text import normalize_whitespace


def test_collapses_all_whitespace_runs():
    assert normalize_whitespace("  a\n\nb\tc   d\n") == "a b c d"


def test_paragraph_styles_become_identical():
    human = "First paragraph.\nSecond paragraph."
    ai = "First paragraph.\n\nSecond paragraph."
    wrapped = "  First\nparagraph.  Second\nparagraph."
    assert normalize_whitespace(human) == normalize_whitespace(ai) == normalize_whitespace(wrapped)


def test_idempotent_and_keeps_non_breaking_content():
    t = "Prices rose 5% (to $3.20) - “quoted”."
    assert normalize_whitespace(t) == t
    assert normalize_whitespace(normalize_whitespace(" x  y ")) == "x y"
