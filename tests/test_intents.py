import pytest

from app.pipeline.intents import NovaTelIntent, classify_intent


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Pourquoi ma facture est plus élevée ce mois-ci ?", NovaTelIntent.BILLING_QUESTION),
        ("Mon paiement par carte est refusé", NovaTelIntent.PAYMENT_ISSUE),
        ("Je veux résilier mon forfait", NovaTelIntent.CANCEL_SUBSCRIPTION),
        ("Comment changer mon offre mobile ?", NovaTelIntent.CHANGE_PLAN),
        ("Je n'ai plus internet depuis ce matin", NovaTelIntent.NETWORK_ISSUE),
        ("J'ai un problème de connexion internet", NovaTelIntent.NETWORK_ISSUE),
        ("Mon téléphone ne capte plus", NovaTelIntent.NETWORK_ISSUE),
        ("Je demande un remboursement", NovaTelIntent.REFUND_REQUEST),
        ("Comment modifier mon adresse ?", NovaTelIntent.ADDRESS_CHANGE),
        ("Est-ce que mon forfait fonctionne en itinérance ?", NovaTelIntent.ROAMING_QUESTION),
        ("Quels forfaits mobiles proposez-vous ?", NovaTelIntent.PLAN_INFORMATION),
        ("Can you guarantee a 70% discount?", NovaTelIntent.PLAN_INFORMATION),
        ("Comment je paris de diriez mon abonnement ?", NovaTelIntent.PLAN_INFORMATION),
        ("Je veux ouvrir une réclamation pour ma facture.", NovaTelIntent.BILLING_QUESTION),
        ("I want to upgrade my mobile plan", NovaTelIntent.CHANGE_PLAN),
        ("Quel est l'état de ma demande ?", NovaTelIntent.REQUEST_STATUS),
        ("Je veux parler à un conseiller", NovaTelIntent.HUMAN_AGENT),
        ("What is the capital of Canada?", NovaTelIntent.OUT_OF_SCOPE),
        ("Pourriez-vous me donner plus de détails ?", NovaTelIntent.OUT_OF_SCOPE),
    ],
)
def test_classifies_business_intents_and_scope(utterance: str, expected: NovaTelIntent) -> None:
    assert classify_intent(utterance) == expected


def test_intent_matching_tolerates_accentless_asr_text() -> None:
    assert classify_intent("Comment resilir mon forfait") == NovaTelIntent.CANCEL_SUBSCRIPTION


@pytest.mark.parametrize(
    "utterance",
    [
        "Comment je peux résélié mon abonement ?",
        "Comment résidiez-vous mon apprum ?",
    ],
)
def test_cancellation_routing_tolerates_common_french_whisper_errors(utterance: str) -> None:
    assert classify_intent(utterance) == NovaTelIntent.CANCEL_SUBSCRIPTION
