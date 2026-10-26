import pytest

from app.pipeline.follow_ups import contextual_retrieval_query, is_contextual_follow_up


@pytest.mark.parametrize(
    "utterance",
    [
        "Pour vivre mon donnis plus de détails ?",
        "Donnez-moi plus d'informations, s'il vous plaît.",
        "Pouvez-vous m'expliquer davantage ?",
        "Et ensuite, que dois-je faire ?",
        "Quel délai faut-il prévoir ?",
        "Quels documents dois-je fournir ?",
        "Ça veut dire quoi ?",
    ],
)
def test_common_follow_up_phrases_are_recognized(utterance: str) -> None:
    assert is_contextual_follow_up(utterance)


def test_new_standalone_customer_question_is_not_mistaken_for_follow_up() -> None:
    assert not is_contextual_follow_up("Où trouver ma facture ?")


def test_follow_up_retrieval_uses_the_last_grounded_topic() -> None:
    previous = "Comment je peux résilier mon abonnement ?"
    follow_up = "Pour vivre mon donnis plus de détails ?"

    query = contextual_retrieval_query(follow_up, previous)

    assert previous in query
    assert follow_up in query
