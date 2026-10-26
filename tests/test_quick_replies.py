import pytest

from app.pipeline.quick_replies import quick_reply


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Bonjour !", "Bonjour !"),
        ("Bonjour, monsieur.", "Bonjour !"),
        ("Salut, ça va ?", "Bonjour ! Merci de demander"),
        ("Bonjour, comment allez-vous ?", "Bonjour ! Merci de demander"),
        ("Cava", "Merci de demander"),
        ("Ça va, merci.", "Tant mieux"),
        ("Je vais bien, merci et vous ?", "Tant mieux"),
        ("Comment allez-vous ?", "Merci de demander"),
        ("Merci beaucoup.", "Avec plaisir"),
        ("Merci beaucoup, c'est gentil.", "Avec plaisir"),
        ("D'accord, merci.", "Très bien"),
        ("Au revoir !", "Au revoir"),
        ("Merci, bonne continuation.", "Avec plaisir. Au revoir"),
        ("Pardon.", "Pas de souci"),
        ("Tu m'entends ?", "Oui, je vous entends"),
        ("Qui êtes-vous ?", "Je suis l’assistant vocal"),
        ("Que pouvez-vous faire ?", "Je peux répondre aux questions"),
        ("Bravo !", "Merci, c’est gentil"),
        ("Hi!", "Bonjour !"),
        ("Thank you.", "Avec plaisir"),
    ],
)
def test_common_social_utterances_receive_a_quick_reply(utterance: str, expected: str) -> None:
    response = quick_reply(utterance)
    assert response is not None
    assert expected in response


def test_repeat_request_uses_previous_assistant_message() -> None:
    response = quick_reply(
        "Pouvez-vous répéter s'il vous plaît ?",
        [{"role": "assistant", "content": "Ouvrez la rubrique Facturation."}],
    )
    assert response == "Bien sûr. Ouvrez la rubrique Facturation."


@pytest.mark.parametrize(
    "utterance",
    [
        "Bonjour, je veux résilier mon abonnement.",
        "Merci, pouvez-vous m'expliquer ma facture ?",
        "Salut. J'ai un problème avec mon paiement.",
        "Comment ça va et où trouver ma facture ?",
        "Je voudrais savoir si vous remboursez les frais, merci.",
        "J'ai besoin d'aide pour changer d'offre.",
    ],
)
def test_mixed_social_and_customer_requests_continue_to_rag(utterance: str) -> None:
    assert quick_reply(utterance) is None


def test_unknown_general_question_is_not_given_an_invented_quick_reply() -> None:
    assert quick_reply("Quel temps fera-t-il demain ?") is None
