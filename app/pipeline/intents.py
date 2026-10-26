"""Deterministic intent routing for NovaTel customer-support conversations."""

import re
import unicodedata
from difflib import SequenceMatcher
from enum import StrEnum


class NovaTelIntent(StrEnum):
    """Supported business intents plus explicit conversation guard categories."""

    BILLING_QUESTION = "billing_question"
    PAYMENT_ISSUE = "payment_issue"
    CANCEL_SUBSCRIPTION = "cancel_subscription"
    CHANGE_PLAN = "change_plan"
    NETWORK_ISSUE = "network_issue"
    REFUND_REQUEST = "refund_request"
    ADDRESS_CHANGE = "address_change"
    ROAMING_QUESTION = "roaming_question"
    PLAN_INFORMATION = "plan_information"
    REQUEST_STATUS = "request_status"
    HUMAN_AGENT = "human_agent"
    SOCIAL = "social"
    OUT_OF_SCOPE = "out_of_scope"


_RULES: tuple[tuple[NovaTelIntent, tuple[str, ...]], ...] = (
    (
        NovaTelIntent.HUMAN_AGENT,
        (
            "conseiller", "agent humain", "personne reelle", "parler a quelqu", "parler a un humain",
            "contacter le service client", "joindre le service client", "appeler le service client",
            "human agent", "human representative", "speak to someone", "contact customer service",
            "call customer service",
        ),
    ),
    (
        NovaTelIntent.REQUEST_STATUS,
        (
            "suivi de ma demande", "etat de ma demande", "ou en est", "statut de ma demande",
            "suivre ma reclamation", "statut de mon remboursement", "ou est mon remboursement",
            "track my request", "request status", "status of my", "where is my refund",
        ),
    ),
    (
        NovaTelIntent.NETWORK_ISSUE,
        (
            "reseau", "internet ne marche", "internet ne fonctionne", "plus internet", "pas de connexion",
            "connexion lente", "connexion internet", "probleme de connexion", "probleme internet", "internet lent",
            "coupure internet", "plus de signal", "pas de signal", "ne capte plus", "panne", "wifi", "wi fi",
            "debit", "network issue", "no internet",
            "internet is down", "connection is slow", "outage",
        ),
    ),
    (NovaTelIntent.REFUND_REQUEST, ("remboursement", "rembourser", "rembourse", "refund", "reimburse")),
    (
        NovaTelIntent.PAYMENT_ISSUE,
        (
            "paiement", "prelevement refuse", "prelevement inconnu", "echec de prelevement", "carte refusee",
            "impaye", "payer", "regler ma facture", "payment", "direct debit", "card declined", "unpaid",
        ),
    ),
    (
        NovaTelIntent.ADDRESS_CHANGE,
        (
            "adresse", "demenagement", "je demenage", "nouvelle adresse", "change my address", "moving house",
            "address change",
        ),
    ),
    (
        NovaTelIntent.ROAMING_QUESTION,
        ("roaming", "itinerance", "etranger", "a l etranger", "hors de france", "international data", "abroad"),
    ),
    (
        NovaTelIntent.CANCEL_SUBSCRIPTION,
        (
            "resili", "resilier", "resiliation", "terminer mon abonnement", "annuler mon forfait", "cancel my",
            "cancel subscription", "terminate my plan",
        ),
    ),
    (
        NovaTelIntent.CHANGE_PLAN,
        (
            "changer d offre", "changer de forfait", "modifier mon forfait", "changer mon abonnement",
            "changer offre", "changer mon offre", "modifier mon offre", "passer a une offre", "change my plan",
            "change my mobile plan", "change my internet plan", "switch plan", "upgrade my plan", "upgrade my",
            "downgrade my plan", "downgrade my", "update my plan",
        ),
    ),
    (
        NovaTelIntent.BILLING_QUESTION,
        (
            "facture", "facturation", "montant preleve", "montant de ma facture", "montant trop eleve",
            "facture elevee", "facture trop elevee", "contester une facture", "billing", "invoice", "charged",
            "bill is higher",
        ),
    ),
    (
        NovaTelIntent.PLAN_INFORMATION,
        (
            "abonnement", "abonnements", "forfait", "offre mobile", "offre internet", "offres", "tarif", "prix",
            "engagement", "data mobile",
            "mobile plan", "internet plan", "pricing", "price", "discount", "remise",
        ),
    ),
)

_CANCEL_WORDS = (
    "resilier", "resilie", "resiliez", "annuler", "annule", "annulez", "terminer", "stopper",
)
_SUBSCRIPTION_WORDS = ("abonnement", "forfait", "contrat", "offre", "ligne")
_POSSESSIVES = {"mon", "ma", "mes"}


def normalize_intent_text(text: str) -> str:
    """Fold accents and punctuation to make rules robust to speech transcripts."""

    folded = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def classify_intent(text: str) -> NovaTelIntent:
    """Choose an intent, tolerating common Whisper misspellings of key terms."""

    normalized = normalize_intent_text(text)
    if _looks_like_cancel_request(normalized):
        return NovaTelIntent.CANCEL_SUBSCRIPTION
    for intent, phrases in _RULES:
        if any(phrase in normalized for phrase in phrases):
            return intent
    return NovaTelIntent.OUT_OF_SCOPE


def _looks_like_cancel_request(normalized: str) -> bool:
    """Recover cancellation requests when Whisper slightly distorts the verb."""

    tokens = normalized.split()
    has_cancel_verb = any(_is_close(token, _CANCEL_WORDS, threshold=0.72) for token in tokens)
    if not has_cancel_verb:
        return False
    has_subscription = any(_is_close(token, _SUBSCRIPTION_WORDS, threshold=0.72) for token in tokens)
    # If ASR corrupts the noun too, a cancellation-like verb plus a possessive
    # object in a question (e.g. "résidiez-vous mon apprum") is still strong evidence.
    return has_subscription or bool(_POSSESSIVES.intersection(tokens))


def _is_close(token: str, candidates: tuple[str, ...], *, threshold: float) -> bool:
    """Compare one ASR token with a short, curated domain vocabulary."""

    if token in candidates:
        return True
    return len(token) >= 5 and any(
        SequenceMatcher(None, token, candidate).ratio() >= threshold for candidate in candidates
    )


def retrieval_hint(intent: NovaTelIntent, query: str = "") -> str:
    """Add domain vocabulary that helps lexical retrieval recover ASR variants."""

    if intent == NovaTelIntent.PLAN_INFORMATION and {
        "abonnement",
        "abonnements",
    }.intersection(normalize_intent_text(query).split()):
        return "abonnement contrat résiliation résilier procédure engagement espace client"
    hints = {
        NovaTelIntent.BILLING_QUESTION: "facture facturation prélèvement montant ligne frais contestation",
        NovaTelIntent.PAYMENT_ISSUE: "paiement prélèvement carte refusé impayé échéance facture",
        NovaTelIntent.CANCEL_SUBSCRIPTION: "résiliation résilier abonnement forfait engagement délai frais procédure",
        NovaTelIntent.CHANGE_PLAN: "changer modifier offre forfait mobile internet migration",
        NovaTelIntent.NETWORK_ISSUE: "réseau panne internet connexion lente diagnostic routeur mobile",
        NovaTelIntent.REFUND_REQUEST: "remboursement rembourser délai moyen de paiement demande",
        NovaTelIntent.ADDRESS_CHANGE: "adresse déménagement modifier profil espace client",
        NovaTelIntent.ROAMING_QUESTION: "itinérance roaming étranger Europe data mobile",
        NovaTelIntent.PLAN_INFORMATION: "offre forfait mobile internet fibre tarif prix données",
        NovaTelIntent.REQUEST_STATUS: "suivi statut état demande réclamation dossier",
        NovaTelIntent.HUMAN_AGENT: "conseiller contact service client escalade assistance",
    }
    return hints.get(intent, "")
