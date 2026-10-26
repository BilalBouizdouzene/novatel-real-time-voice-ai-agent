"""Refresh attributed public consumer-information pages in the local KB.

This is a small allow-list scraper, not a general-purpose crawler. It fetches
only the official pages declared below, extracts readable HTML text, and saves
Markdown with source URL and retrieval timestamp. No credentials or user data
are sent; requests use a descriptive user-agent and a bounded response size.
"""

import argparse
import logging
import re
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "knowledge_base" / "public_guidance"
MAX_RESPONSE_BYTES = 4_000_000
MAX_DOCUMENT_CHARS = 18_000
USER_AGENT = "RealTimeVoiceAIAgent-KB-Refresh/1.0 (official-source allow-list; polite fetching)"
ALLOWED_HOSTS = {"www.economie.gouv.fr", "www.arcep.fr", "www.service-public.fr", "www.service-public.gouv.fr"}

SOURCES: dict[str, tuple[str, str]] = {
    "telecom_subscriptions": (
        "Abonnements téléphoniques et internet : analyser l’offre des opérateurs",
        "https://www.economie.gouv.fr/dgccrf/les-fiches-pratiques/abonnements-telephoniques-et-internet-comment-bien-analyser-loffre-des-operateurs",
    ),
    "online_termination": (
        "Résilier ses contrats conclus sur internet",
        "https://www.economie.gouv.fr/dgccrf/actualites-dgccrf/resilier-ses-contrats-conclus-sur-internet-est-desormais-plus-facile",
    ),
    "withdrawal_refund": (
        "Vente à distance : droit de rétractation et remboursement",
        "https://www.economie.gouv.fr/particuliers/mes-droits-conso/bien-consommer/vente-distance-tout-savoir-sur-votre-droit-de-retractation",
    ),
    "direct_debit": (
        "Prélèvement bancaire : contestation et délais",
        "https://www.service-public.fr/particuliers/vosdroits/F20752",
    ),
    "telecom_complaints": (
        "Résoudre un litige avec un opérateur télécom",
        "https://www.arcep.fr/mes-demarches-et-services/consommateurs/fiches-pratiques/particuliers-comment-resoudre-un-litige-avec-votre-operateur.html",
    ),
    "online_disputes": (
        "Achats et services en ligne : conseils en cas de litige",
        "https://www.economie.gouv.fr/particuliers/mes-droits-conso/gerer-un-litige/achats-et-services-en-ligne-6-conseils-en-cas-de-litige",
    ),
}

_SKIP_TAGS = {"script", "style", "svg", "nav", "header", "footer", "aside", "form", "button", "noscript"}
_BLOCK_TAGS = {"h1", "h2", "h3", "p", "li"}
_BOILERPLATE_EXACT = {
    "aller au contenu",
    "aller au menu",
    "république française",
    "se connecter",
    "partager la page",
    "facebook",
    "x",
    "linkedin",
    "courriel",
    "lien copié",
    "ce sujet vous intéresse ?",
}
_BOILERPLATE_PARTS = (
    "ajoutez cette page à vos favoris",
    "a été ajouté à vos favoris",
    "vous recevrez un courriel dès que la page",
    "votre abonnement n’a pas pu être pris en compte",
    "votre abonnement n'a pas pu être pris en compte",
    "vous devez vous connecter à votre espace personnel afin de vous abonner",
    "titlecontent",
)


class ArticleTextParser(HTMLParser):
    """Extract article headings and prose without scripts or navigation text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._main_blocks: list[tuple[str, str]] = []
        self._main_depth = 0
        self._skip_depth = 0
        self._block_tag: str | None = None
        self._block_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "main":
            self._main_depth += 1
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and self._block_tag is None and tag in _BLOCK_TAGS:
            self._block_tag = tag
            self._block_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == self._block_tag:
            text = re.sub(r"\s+", " ", " ".join(self._block_parts)).strip()
            lowered = text.casefold()
            boilerplate = lowered in _BOILERPLATE_EXACT or any(part in lowered for part in _BOILERPLATE_PARTS)
            if text and not boilerplate:
                block = (tag, text)
                self.blocks.append(block)
                if self._main_depth:
                    self._main_blocks.append(block)
            self._block_tag = None
            self._block_parts = []
        if tag == "main" and self._main_depth:
            self._main_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._block_tag is not None:
            self._block_parts.append(data)

    def article_text(self) -> tuple[str, str]:
        blocks = self._main_blocks or self.blocks
        title = next((text for tag, text in blocks if tag == "h1"), "")
        seen: set[str] = set()
        lines: list[str] = []
        for tag, text in blocks:
            if tag == "h1" or text in seen:
                continue
            seen.add(text)
            lines.append(f"## {text}" if tag in {"h2", "h3"} else text)
        return title, "\n\n".join(lines)


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Refusing non-allow-listed source URL: {url}")


def _fetch_page(url: str, timeout: float = 20.0) -> tuple[str, str]:
    """Fetch a bounded HTML page and return its final URL and decoded body."""

    _validate_url(url)
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS allow-list above
        final_url = response.geturl()
        _validate_url(final_url)
        content_type = response.headers.get_content_type()
        if content_type != "text/html":
            raise ValueError(f"Expected HTML, received {content_type}")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError(f"Page exceeds the {MAX_RESPONSE_BYTES}-byte safety limit")
    return final_url, raw.decode("utf-8", errors="replace")


def scrape_source(slug: str, label: str, url: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Download one official page and write an attributed, bounded Markdown copy."""

    final_url, html = _fetch_page(url)
    parser = ArticleTextParser()
    parser.feed(html)
    title, content = parser.article_text()
    if len(content) < 300:
        raise ValueError(f"Could not extract enough article text from {final_url}")
    content = content[:MAX_DOCUMENT_CHARS].rsplit(" ", maxsplit=1)[0]
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    document = (
        f"# {title or label}\n\n"
        "Source officielle consultée automatiquement. Résumé/extrait informatif : vérifier la page et les conditions "
        "propres au contrat avant toute décision.\n\n"
        f"URL source : {final_url}\n\n"
        f"Dernière collecte : {timestamp}\n\n"
        f"{content}\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{slug}.md"
    path.write_text(document, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["all", *SOURCES],
        default="all",
        help="Refresh all allow-listed pages or just one source.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    selected = list(SOURCES.items()) if args.only == "all" else [(args.only, SOURCES[args.only])]
    failures = 0
    for index, (slug, (label, url)) in enumerate(selected):
        try:
            path = scrape_source(slug, label, url)
            character_count = len(path.read_text(encoding="utf-8"))
            logging.info("Saved %s (%d characters)", path.relative_to(PROJECT_ROOT), character_count)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            failures += 1
            logging.error("Could not refresh %s: %s", slug, exc)
        if index < len(selected) - 1:
            time.sleep(1.0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
