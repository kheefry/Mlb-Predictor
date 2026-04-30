"""Robust player-name matching across data sources.

MLB Stats API uses accented names ('José Ramírez') while Bovada strips
accents ('Jose Ramirez'). Suffixes ('Jr.', 'Sr.', 'III'), middle names, and
Latino multi-name conventions also vary. We canonicalize aggressively then
fall back to fuzzy matching on the surname when first-name initials match.
"""
from __future__ import annotations
import re
import unicodedata
from typing import Iterable, Optional


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


_SUFFIXES = re.compile(r"\b(Jr\.?|Sr\.?|II|III|IV)\b", re.I)
_PUNCT = re.compile(r"[^\w\s]")


def canonical(name: str) -> str:
    """Lower-case, accent-stripped, suffix-stripped, single-spaced."""
    if not name:
        return ""
    n = _strip_accents(name)
    n = _SUFFIXES.sub("", n)
    n = _PUNCT.sub("", n)
    return " ".join(n.lower().split())


def split_first_last(canon: str) -> tuple[str, str]:
    parts = canon.split()
    if not parts:
        return "", ""
    return parts[0], " ".join(parts[1:]) or parts[0]


def find_match(target: str, pool: Iterable[str]) -> Optional[str]:
    """Return the best match from `pool` for `target`, or None.

    Matching priority:
      1. Exact canonical match
      2. Same surname AND first-name initial matches
      3. Same surname uniquely
    """
    t = canonical(target)
    if not t:
        return None
    pool_list = list(pool)
    canons = {p: canonical(p) for p in pool_list}

    # 1. Exact
    for orig, c in canons.items():
        if c == t:
            return orig

    # 2. Surname AND first-initial both match. We never accept a surname-only
    # match — e.g. "Yordan Alvarez" must NOT match "Francisco Alvarez" just
    # because they share a surname. Common Latino surnames (Garcia, Hernandez,
    # Rodriguez, Alvarez, Lopez) make surname-only matching dangerous.
    t_first, t_last = split_first_last(t)
    if t_first:
        candidates = [orig for orig, c in canons.items()
                      if split_first_last(c)[1] == t_last
                      and split_first_last(c)[0][:1] == t_first[:1]]
        if len(candidates) == 1:
            return candidates[0]

    # 3. Multi-token surname overlap: "Salvador Perez" -> "Salvador Perez Jr."
    if t_first:
        for orig, c in canons.items():
            c_first, c_last = split_first_last(c)
            if c_first == t_first and (t_last in c_last or c_last in t_last):
                return orig

    return None
