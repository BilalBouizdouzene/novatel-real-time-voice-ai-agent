import pytest

from scripts.scrape_knowledge_base import ArticleTextParser, _validate_url


def test_article_parser_keeps_prose_and_headings_and_drops_navigation() -> None:
    parser = ArticleTextParser()
    parser.feed(
        """
        <nav><p>Menu à ignorer</p></nav>
        <header><p>République Française</p></header>
        <main><h1>Résiliation</h1><h2>Préavis</h2>
        <p>Un préavis peut être prévu au contrat.</p>
        <p>Vous recevrez un courriel dès que la page sera mise à jour.</p>
        <p>Le créancier : titleContent doit être identifié.</p>
        <ul><li>Conserver la confirmation.</li></ul>
        <script><p>Ne pas inclure</p></script></main>
        <p>Contenu hors article à exclure.</p>
        """
    )

    title, content = parser.article_text()

    assert title == "Résiliation"
    assert "## Préavis" in content
    assert "Conserver la confirmation." in content
    assert "Menu à ignorer" not in content
    assert "République Française" not in content
    assert "Contenu hors article" not in content
    assert "titleContent" not in content
    assert "Ne pas inclure" not in content


@pytest.mark.parametrize(
    "url",
    [
        "http://www.economie.gouv.fr/article",
        "https://example.com/article",
        "https://127.0.0.1/private",
    ],
)
def test_scraper_rejects_urls_outside_https_allowlist(url: str) -> None:
    with pytest.raises(ValueError, match="allow-listed"):
        _validate_url(url)
