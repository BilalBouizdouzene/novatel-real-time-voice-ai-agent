"""Recognize short follow-up turns and anchor their retrieval to the prior topic."""

import re
import unicodedata


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


_FOLLOW_UP_PATTERNS = (
    re.compile(r"\bplus\s+(?:de|du|des|d)\s+(?:detail\w*|precision\w*|information\w*)\b"),
    re.compile(r"\b(?:more details|more information|tell me more)\b"),
    re.compile(r"\b(?:explique|expliquez|precise|precisez|detaille|detaillez|developpe|developpez)\b"),
    re.compile(r"\b(?:en savoir plus|davantage|plus clairement|approfondir)\b"),
    re.compile(r"\b(?:et ensuite|et apres|apres ca|dans ce cas|et si)\b"),
    re.compile(r"\b(?:quel delai|quelle duree|combien de temps|quels documents|quelle etape|quelles etapes)\b"),
    re.compile(r"\b(?:ca veut dire quoi|cela veut dire quoi|c est a dire|pourquoi)\b"),
)


def is_contextual_follow_up(text: str) -> bool:
    """Return true for a request whose subject is likely established by prior turns."""

    normalized = _normalize(text)
    return any(pattern.search(normalized) for pattern in _FOLLOW_UP_PATTERNS)


def contextual_retrieval_query(text: str, previous_query: str | None) -> str:
    """Add the last grounded topic to a short follow-up without changing its wording."""

    if previous_query and is_contextual_follow_up(text):
        return f"{previous_query}\n{text}"
    return text
