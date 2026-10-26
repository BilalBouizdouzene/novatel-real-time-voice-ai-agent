"""Deterministic handling for common social turns in a voice conversation."""

import re
import unicodedata

_INTENT_PHRASES: dict[str, set[str]] = {
    "thanks": {
        "merci",
        "merci beaucoup",
        "merci bien",
        "merci infiniment",
        "merci pour tout",
        "merci pour votre aide",
        "merci pour ton aide",
        "je vous remercie",
        "je vous remercie beaucoup",
        "je te remercie",
        "c est gentil merci",
        "merci c est gentil",
        "merci beaucoup c est gentil",
        "merci c est tres gentil",
        "c est tres gentil merci",
        "thanks",
        "thank you",
        "thank you very much",
    },
    "greeting": {
        "bonjour",
        "bonsoir",
        "salut",
        "coucou",
        "hello",
        "hi",
        "hey",
        "bonjour assistant",
        "bonjour tout le monde",
        "bonjour a tous",
        "bonjour monsieur",
        "bonjour madame",
        "bonjour mademoiselle",
        "salut assistant",
        "salut tout le monde",
    },
    "wellbeing": {
        "ca va",
        "cava",
        "sa va",
        "comment ca va",
        "comment sa va",
        "comment allez vous",
        "comment vas tu",
        "tu vas bien",
        "vous allez bien",
        "est ce que ca va",
        "et toi ca va",
        "et vous ca va",
        "ca va bien et toi",
        "ca va bien et vous",
        "ca va toi",
        "ca va vous",
        "ca va et toi",
        "ca va et vous",
        "pas mal et toi",
        "pas mal et vous",
        "ca roule",
        "how are you",
        "how are you doing",
    },
    "wellbeing_statement": {
        "ca va merci",
        "je vais bien",
        "je vais bien merci",
        "je vais bien merci et toi",
        "je vais bien merci et vous",
        "bien merci et toi",
        "bien merci et vous",
    },
    "farewell": {
        "au revoir",
        "a bientot",
        "a demain",
        "a tout a l heure",
        "a plus",
        "bonne journee",
        "bonne soiree",
        "bonne nuit",
        "bonne continuation",
        "a plus tard",
        "a la prochaine",
        "bye",
        "goodbye",
        "see you",
    },
    "agreement": {
        "oui",
        "oui merci",
        "d accord",
        "ok",
        "okay",
        "ca marche",
        "parfait",
        "tres bien",
        "compris",
        "entendu",
        "exactement",
        "yes",
        "all right",
    },
    "decline": {"non", "non merci", "pas besoin", "ce sera tout", "no", "no thanks"},
    "apology": {"pardon", "excusez moi", "excuse moi", "desole", "desolee", "sorry"},
    "repeat": {
        "repete",
        "repetez",
        "pouvez vous repeter",
        "peux tu repeter",
        "je n ai pas compris",
        "j ai pas compris",
        "je nai pas compris",
        "je n ai pas bien compris",
        "pardon pouvez vous repeter",
        "i did not understand",
        "could you repeat",
    },
    "identity": {
        "qui es tu",
        "qui etes vous",
        "comment tu t appelles",
        "comment vous appelez vous",
        "quel est ton nom",
        "tu es un robot",
        "tu es une intelligence artificielle",
        "are you a robot",
        "who are you",
    },
    "capabilities": {
        "que peux tu faire",
        "que pouvez vous faire",
        "en quoi peux tu m aider",
        "en quoi pouvez vous m aider",
        "comment peux tu m aider",
        "comment pouvez vous m aider",
        "quelles sont tes capacites",
        "aide moi",
        "aidez moi",
        "what can you do",
    },
    "availability": {
        "tu es la",
        "vous etes la",
        "est ce que tu es la",
        "est ce que vous etes la",
        "tu m entends",
        "vous m entendez",
        "are you there",
    },
    "compliment": {
        "bravo",
        "super",
        "genial",
        "c est parfait",
        "tu es gentil",
        "vous etes gentil",
        "tu es geniale",
        "vous etes geniale",
        "you are great",
    },
    "smalltalk": {
        "quoi de neuf",
        "comment se passe ta journee",
        "comment se passe votre journee",
        "tu fais quoi",
        "que fais tu",
        "what is new",
    },
}

_REPLIES = {
    "thanks": "Avec plaisir. N’hésitez pas si vous avez une autre question.",
    "greeting": (
        "Bonjour ! Je suis Nova, l’assistante NovaTel. Je peux vous aider avec vos forfaits, factures "
        "ou services télécom."
    ),
    "wellbeing": "Merci de demander ! Je suis prêt à vous aider. Que puis-je faire pour vous ?",
    "wellbeing_statement": "Tant mieux ! Que puis-je faire pour vous aujourd’hui ?",
    "farewell": "Au revoir, et bonne journée.",
    "agreement": "Très bien. Je reste à votre écoute si vous avez besoin d’autre chose.",
    "decline": "D’accord. Je reste à votre disposition si vous avez besoin d’aide plus tard.",
    "apology": "Pas de souci. Je vous écoute.",
    "identity": (
        "Je suis l’assistant vocal NovaTel. Je m’appelle Nova et je peux vous renseigner sur les services télécom "
        "décrits dans ma base de connaissances."
    ),
    "capabilities": (
        "Je peux répondre aux questions de ma base NovaTel, notamment sur les forfaits, la facturation, les paiements, "
        "le réseau, l’itinérance et les demandes de remboursement."
    ),
    "availability": "Oui, je vous entends. Dites-moi ce dont vous avez besoin.",
    "compliment": "Merci, c’est gentil ! Je suis là pour vous aider.",
    "smalltalk": (
        "Je suis prête à vous aider pour un service NovaTel. Votre question concerne-t-elle un forfait, une facture "
        "ou votre connexion ?"
    ),
}

# These conversational openers may precede a social phrase. If meaningful text
# follows (for example, a service request), matching still fails and RAG runs.
_POLITE_OPENERS = ("bonjour", "bonsoir", "salut", "coucou", "hello", "hi", "hey", "alors", "euh", "et")
_POLITE_SUFFIXES = ("s il vous plait", "s il te plait", "svp", "stp", "please")


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


_PHRASE_TO_INTENT = {phrase: intent for intent, phrases in _INTENT_PHRASES.items() for phrase in phrases}
_KNOWN_PHRASES = sorted(_PHRASE_TO_INTENT, key=len, reverse=True)


def _match_intent(normalized: str) -> list[str] | None:
    """Match complete social phrases, including simple greetings and politeness."""

    if normalized in _PHRASE_TO_INTENT:
        return [_PHRASE_TO_INTENT[normalized]]
    for opener in _POLITE_OPENERS:
        if normalized.startswith(f"{opener} "):
            intents = _match_intent(normalized[len(opener) :].strip())
            if intents:
                opener_intent = _PHRASE_TO_INTENT.get(opener)
                return [opener_intent, *intents] if opener_intent else intents
    for suffix in _POLITE_SUFFIXES:
        if normalized.endswith(f" {suffix}"):
            intents = _match_intent(normalized[: -len(suffix)].strip())
            if intents:
                return intents
    for phrase in _KNOWN_PHRASES:
        if normalized.startswith(f"{phrase} "):
            remainder = _match_intent(normalized[len(phrase) :].strip())
            if remainder:
                return [_PHRASE_TO_INTENT[phrase], *remainder]
    return None


def _intent_for_utterance(text: str) -> list[str] | None:
    """Return intents only when every sentence is recognized as social speech."""

    sentences = [part.strip() for part in re.split(r"[.!?;]+", text) if part.strip()]
    if not sentences:
        sentences = [text]
    intents: list[str] = []
    for sentence in sentences:
        sentence_intents = _match_intent(_normalize(sentence))
        if sentence_intents is None:
            return None
        intents.extend(sentence_intents)
    return intents


def quick_reply(text: str, history: list[dict[str, str]] | None = None) -> str | None:
    """Handle common social turns without RAG/LLM, preserving real requests.

    The matcher intentionally has a bounded scope: an utterance is handled here
    only when all of its sentences match known social intents. Unknown or mixed
    requests continue through the normal knowledge-grounded pipeline.
    """

    intents = _intent_for_utterance(text)
    if not intents:
        return None
    if "repeat" in intents:
        previous_answer = next(
            (message["content"] for message in reversed(history or []) if message.get("role") == "assistant"),
            None,
        )
        if previous_answer:
            return f"Bien sûr. {previous_answer}"
        return "Bien sûr. Quelle partie souhaitez-vous que je répète ?"
    if "farewell" in intents and "thanks" in intents:
        return "Avec plaisir. Au revoir, et bonne journée."
    if "greeting" in intents and "wellbeing" in intents:
        return "Bonjour ! Merci de demander, je suis prêt à vous aider. Que puis-je faire pour vous ?"
    if "wellbeing_statement" in intents and "wellbeing" in intents:
        return "Tant mieux, merci de demander ! Que puis-je faire pour vous aujourd’hui ?"
    if "agreement" in intents and "thanks" in intents:
        return "Avec plaisir. Très bien, je reste à votre écoute."
    if "wellbeing" in intents and "thanks" in intents:
        return "Tant mieux ! Que puis-je faire pour vous aujourd’hui ?"
    return _REPLIES[intents[-1]]
