"""Utilities for normalizing Hebrew text with hebspacy.

This module provides small helper functions for cleaning Hebrew strings,
including removal of niqqud (vowel marks), unification of final letters,
and optional lemmatization via a hebspacy pipeline.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List, Optional

try:  # hebspacy depends on spaCy; delay import so the module is optional.
    import hebspacy
    from spacy.language import Language
except ImportError:  # pragma: no cover - optional dependency.
    hebspacy = None
    Language = None  # type: ignore

NIQQUD_PATTERN = re.compile(r"[\u0591-\u05C7]")
FINAL_LETTERS_MAP = str.maketrans({
    "ך": "כ",
    "ם": "מ",
    "ן": "נ",
    "ף": "פ",
    "ץ": "צ",
})


def remove_niqqud(text: str) -> str:
    """Strip Hebrew vowel marks (niqqud) from *text*."""
    return NIQQUD_PATTERN.sub("", text)


def unify_final_letters(text: str) -> str:
    """Map final Hebrew letters to their standard forms (ך→כ, ם→מ, ...)."""
    return text.translate(FINAL_LETTERS_MAP)


@lru_cache(maxsize=2)
def load_hebrew_nlp(model_name: str = "he_core_news_trf") -> "Language":
    """Load and cache a hebspacy pipeline for lemmatization.

    Parameters
    ----------
    model_name: str
        Name passed to ``hebspacy.load``. The default corresponds to the
        transformer-based pipeline; adjust if you need a lighter model.

    Raises
    ------
    ImportError
        If hebspacy (and spaCy) are not available in the environment.
    """
    if hebspacy is None:
        raise ImportError(
            "hebspacy is required for lemmatization. Install it via `pip install hebspacy`."
        )
    return hebspacy.load(model_name)


def normalize_text(
    text: str,
    *,
    nlp: Optional["Language"] = None,
    remove_vowels: bool = True,
    unify_finals: bool = True,
    lemmatize: bool = True,
) -> str:
    """Normalize a single Hebrew string.

    Parameters
    ----------
    text: str
        The original string to normalise.
    nlp: Optional[Language]
        A hebspacy pipeline. If omitted and ``lemmatize`` is True, the default
        pipeline is loaded via :func:`load_hebrew_nlp`.
    remove_vowels: bool
        Strip niqqud marks.
    unify_finals: bool
        Map final letter forms to their standard shapes.
    lemmatize: bool
        Replace tokens with their lemmas using the provided pipeline.
    """
    if not text:
        return ""

    normalised = text
    if remove_vowels:
        normalised = remove_niqqud(normalised)
    if unify_finals:
        normalised = unify_final_letters(normalised)

    if lemmatize:
        pipeline = nlp or load_hebrew_nlp()
        doc = pipeline(normalised)
        lemmas: List[str] = []
        for token in doc:
            lemma = token.lemma_ or token.text
            lemmas.append(lemma)
        normalised = " ".join(lemmas)
        if remove_vowels:
            normalised = remove_niqqud(normalised)
        if unify_finals:
            normalised = unify_final_letters(normalised)

    return normalised.strip()


def normalize_batch(
    texts: Iterable[str],
    *,
    nlp: Optional["Language"] = None,
    remove_vowels: bool = True,
    unify_finals: bool = True,
    lemmatize: bool = True,
) -> List[str]:
    """Normalise an iterable of Hebrew strings using the same configuration."""
    pipeline = nlp
    if lemmatize and pipeline is None:
        pipeline = load_hebrew_nlp()
    return [
        normalize_text(
            text,
            nlp=pipeline,
            remove_vowels=remove_vowels,
            unify_finals=unify_finals,
            lemmatize=lemmatize,
        )
        for text in texts
    ]

__all__ = [
    "normalize_text",
    "normalize_batch",
    "remove_niqqud",
    "unify_final_letters",
    "load_hebrew_nlp",
]
