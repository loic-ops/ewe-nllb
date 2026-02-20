"""Scraper Wikipedia Éwé pour données monolingues."""

import re
from pathlib import Path

from tqdm import tqdm

from scrapers.base_scraper import BaseScraper
from config import WIKI_API_URL


class WikipediaEweScraper(BaseScraper):
    """
    Extrait le texte monolingue Éwé depuis ee.wikipedia.org
    via l'API MediaWiki. Données monolingues pour apprendre
    la structure profonde de la langue.
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, delay=1.0)

    def _get_all_page_ids(self) -> list[int]:
        """Énumère tous les IDs d'articles via l'API allpages."""
        page_ids = []
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": 500,
            "apnamespace": 0,  # namespace principal uniquement
            "format": "json",
        }

        self.logger.info("Énumération des pages Wikipedia Éwé...")
        continuation = None

        while True:
            request_params = dict(params)
            if continuation:
                request_params["apcontinue"] = continuation

            self._rate_limit()
            response = self.session.get(
                WIKI_API_URL, params=request_params, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("allpages", [])
            page_ids.extend(p["pageid"] for p in pages)

            if "continue" not in data:
                break
            continuation = data["continue"]["apcontinue"]

        self.logger.info(f"  {len(page_ids)} pages trouvées")
        return page_ids

    def _get_pages_text(self, page_ids: list[int]) -> list[dict]:
        """Récupère le texte brut pour un lot de pages."""
        params = {
            "action": "query",
            "pageids": "|".join(str(pid) for pid in page_ids),
            "prop": "extracts",
            "explaintext": "true",
            "format": "json",
        }

        self._rate_limit()
        response = self.session.get(WIKI_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        records = []
        for page in data.get("query", {}).get("pages", {}).values():
            extract = page.get("extract", "").strip()
            title = page.get("title", "")

            if not extract or len(extract) < 50:
                continue

            # Découper en phrases
            sentences = self._split_sentences(extract)
            for sentence in sentences:
                if len(sentence) >= 10:
                    records.append({
                        "ee": sentence,
                        "source": "wikipedia_ee",
                        "title": title,
                    })

        return records

    def _split_sentences(self, text: str) -> list[str]:
        """Découpe le texte en phrases individuelles."""
        # Supprimer les titres de sections (== Titre ==)
        text = re.sub(r"={2,}.*?={2,}", "", text)

        # Découper sur les points, points d'interrogation, points d'exclamation
        sentences = re.split(r"(?<=[.!?])\s+", text)

        result = []
        for s in sentences:
            s = s.strip()
            # Filtrer les lignes trop courtes ou qui ne sont que des chiffres/symboles
            if len(s) >= 10 and re.search(r"[a-zA-ZɖɛƒŋɔƉƐƑŊƆ]", s):
                result.append(s)

        return result

    def collect(self) -> list[dict]:
        self.logger.info("Scraping Wikipedia Éwé (données monolingues)...")

        page_ids = self._get_all_page_ids()
        records = []

        # Traiter par lots de 50 (limite API MediaWiki)
        batch_size = 50
        for i in tqdm(
            range(0, len(page_ids), batch_size),
            desc="Wikipedia Éwé",
            total=len(page_ids) // batch_size + 1,
        ):
            batch = page_ids[i : i + batch_size]
            try:
                records.extend(self._get_pages_text(batch))
            except Exception as e:
                self.logger.warning(f"Erreur batch {i}: {e}")

        self.logger.info(f"Wikipedia Éwé: {len(records)} phrases monolingues")
        self.save_records(records)
        return records
