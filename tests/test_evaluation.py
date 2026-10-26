from app.speech.evaluation import character_error_rate, word_error_rate


def test_reference_metrics_cover_exact_and_mismatched_text() -> None:
    assert word_error_rate("bonjour le monde", "bonjour le monde") == 0.0
    assert word_error_rate("bonjour le monde", "bonjour monde") == 1 / 3
    assert character_error_rate("abc", "abc") == 0.0
    assert character_error_rate("abc", "adc") == 1 / 3
