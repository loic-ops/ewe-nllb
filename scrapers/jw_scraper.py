"""Scraper JW.org : Bible parallèle (fr-ee) et articles bilingues."""

from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

from scrapers.base_scraper import BaseScraper
from config import JW_EWE_CONFIG, JW_FR_CONFIG

# Livres de la Bible avec nombre de chapitres
BIBLE_BOOKS = [
    ("Genèse", 50), ("Exode", 40), ("Lévitique", 27), ("Nombres", 36),
    ("Deutéronome", 34), ("Josué", 24), ("Juges", 21), ("Ruth", 4),
    ("1 Samuel", 31), ("2 Samuel", 24), ("1 Rois", 22), ("2 Rois", 25),
    ("1 Chroniques", 29), ("2 Chroniques", 36), ("Esdras", 10),
    ("Néhémie", 13), ("Esther", 10), ("Job", 42), ("Psaumes", 150),
    ("Proverbes", 31), ("Ecclésiaste", 12), ("Cantique", 8),
    ("Isaïe", 66), ("Jérémie", 52), ("Lamentations", 5),
    ("Ézéchiel", 48), ("Daniel", 12), ("Osée", 14), ("Joël", 3),
    ("Amos", 9), ("Abdias", 1), ("Jonas", 4), ("Michée", 7),
    ("Nahoum", 3), ("Habacuc", 3), ("Sophonie", 3), ("Aggée", 2),
    ("Zacharie", 14), ("Malachie", 4),
    # Nouveau Testament
    ("Matthieu", 28), ("Marc", 16), ("Luc", 24), ("Jean", 21),
    ("Actes", 28), ("Romains", 16), ("1 Corinthiens", 16),
    ("2 Corinthiens", 13), ("Galates", 6), ("Éphésiens", 6),
    ("Philippiens", 4), ("Colossiens", 4), ("1 Thessaloniciens", 5),
    ("2 Thessaloniciens", 3), ("1 Timothée", 6), ("2 Timothée", 4),
    ("Tite", 3), ("Philémon", 1), ("Hébreux", 13), ("Jacques", 5),
    ("1 Pierre", 5), ("2 Pierre", 3), ("1 Jean", 5), ("2 Jean", 1),
    ("3 Jean", 1), ("Jude", 1), ("Révélation", 22),
]


class JWBibleScraper(BaseScraper):
    """
    Scrape la Bible NWT depuis wol.jw.org en Éwé et Français.
    Alignement par livre/chapitre/verset.
    ~31 000 paires de versets attendues.
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, delay=2.0)

    def _get_chapter_url(self, lang_config: dict, book_num: int, chapter: int) -> str:
        iface = lang_config["iface"]
        region = lang_config["region"]
        lang_pref = lang_config["lang_pref"]
        return (
            f"https://wol.jw.org/{iface}/wol/b/"
            f"{region}/{lang_pref}/nwt/{book_num}/{chapter}"
        )

    def _extract_verses(self, html: str) -> dict[int, str]:
        """Extrait les versets d'une page de chapitre Bible."""
        soup = BeautifulSoup(html, "html.parser")
        verses = {}

        # Stratégie 1 : éléments avec attribut data-pid (numéro de paragraphe/verset)
        for elem in soup.select("[data-pid]"):
            try:
                verse_num = int(elem.get("data-pid", 0))
            except (ValueError, TypeError):
                continue
            text = elem.get_text(strip=True)
            # Nettoyer les numéros de versets en début de texte
            if text and verse_num > 0:
                # Supprimer le numéro du verset s'il est au début
                text = text.lstrip("0123456789 \xa0+*")
                if text:
                    verses[verse_num] = text

        # Stratégie 2 (fallback) : chercher les spans de versets
        if not verses:
            for span in soup.select("span.v"):
                verse_id = span.get("id", "")
                try:
                    verse_num = int(verse_id.split("-")[-1]) if "-" in verse_id else 0
                except (ValueError, IndexError):
                    continue
                text = span.get_text(strip=True)
                if text and verse_num > 0:
                    text = text.lstrip("0123456789 \xa0+*")
                    if text:
                        verses[verse_num] = text

        return verses

    def collect(self) -> list[dict]:
        self.logger.info("Scraping Bible JW.org (Éwé + Français)...")
        records = []
        consecutive_errors = 0
        max_consecutive_errors = 5  # Arrêter après 5 erreurs d'affilée

        # Charger le checkpoint si existant
        checkpoint = self.load_checkpoint() or {"last_book": 0, "last_chapter": 0}
        start_book = checkpoint.get("last_book", 0)

        for book_num, (book_name, num_chapters) in enumerate(
            tqdm(BIBLE_BOOKS, desc="Bible JW.org"), 1
        ):
            if book_num < start_book:
                continue

            if consecutive_errors >= max_consecutive_errors:
                self.logger.warning(
                    f"Arrêt après {max_consecutive_errors} erreurs consécutives. "
                    f"Sauvegarde de {len(records)} paires collectées."
                )
                break

            for chapter in range(1, num_chapters + 1):
                if book_num == start_book and chapter <= checkpoint.get("last_chapter", 0):
                    continue

                if consecutive_errors >= max_consecutive_errors:
                    break

                try:
                    ee_url = self._get_chapter_url(JW_EWE_CONFIG, book_num, chapter)
                    fr_url = self._get_chapter_url(JW_FR_CONFIG, book_num, chapter)

                    ee_html = self._fetch(ee_url).text
                    fr_html = self._fetch(fr_url).text

                    ee_verses = self._extract_verses(ee_html)
                    fr_verses = self._extract_verses(fr_html)

                    for verse_num in ee_verses:
                        if verse_num in fr_verses:
                            records.append({
                                "fr": fr_verses[verse_num],
                                "ee": ee_verses[verse_num],
                                "source": "jw_bible",
                                "ref": f"{book_name} {chapter}:{verse_num}",
                            })

                    consecutive_errors = 0  # Reset si succès

                except Exception as e:
                    consecutive_errors += 1
                    self.logger.warning(
                        f"Erreur {book_name} {chapter} ({consecutive_errors}/{max_consecutive_errors}): {e}"
                    )

                # Sauvegarder le checkpoint régulièrement
                if chapter % 5 == 0 or consecutive_errors > 0:
                    self.save_checkpoint({
                        "last_book": book_num,
                        "last_chapter": chapter,
                    })
                    # Sauvegarder aussi les données déjà collectées
                    if records:
                        self.save_records(records, "bible.jsonl")

        self.logger.info(f"Bible JW.org: {len(records)} paires de versets")
        if records:
            self.save_records(records, "bible.jsonl")
        return records


class JWArticleScraper(BaseScraper):
    """
    Scrape les articles bilingues de JW.org (Watchtower, Awake!, etc.).
    Alignement au niveau paragraphe.
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, delay=2.5)

    def _get_publication_urls(self, lang_config: dict) -> list[str]:
        """Récupère les URLs des articles depuis l'index des publications."""
        iface = lang_config["iface"]
        region = lang_config["region"]
        lang_pref = lang_config["lang_pref"]
        base = f"https://wol.jw.org/{iface}/wol/library/{region}/{lang_pref}"

        article_urls = []
        try:
            response = self._fetch(base)
            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.select("a[href*='/wol/d/']"):
                href = link.get("href", "")
                if href:
                    full_url = f"https://wol.jw.org{href}" if href.startswith("/") else href
                    article_urls.append(full_url)

        except Exception as e:
            self.logger.error(f"Erreur récupération index: {e}")

        return article_urls

    def _extract_paragraphs(self, html: str) -> list[str]:
        """Extrait les paragraphes du corps d'un article."""
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = []

        # Chercher le contenu principal de l'article
        for selector in ["article p", ".bodyTxt p", "#article p", ".docSubContent p"]:
            elements = soup.select(selector)
            if elements:
                for p in elements:
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:
                        paragraphs.append(text)
                break

        return paragraphs

    def _convert_url_to_other_lang(
        self, url: str, from_config: dict, to_config: dict
    ) -> str:
        """Convertit une URL d'une langue à l'autre."""
        result = url
        result = result.replace(
            f"/{from_config['iface']}/", f"/{to_config['iface']}/"
        )
        result = result.replace(
            f"/{from_config['region']}/", f"/{to_config['region']}/"
        )
        result = result.replace(
            f"/{from_config['lang_pref']}/", f"/{to_config['lang_pref']}/"
        )
        return result

    def collect(self) -> list[dict]:
        self.logger.info("Scraping articles JW.org (Éwé + Français)...")
        records = []
        consecutive_errors = 0
        max_consecutive_errors = 5

        # Récupérer les URLs des articles en Éwé
        ee_urls = self._get_publication_urls(JW_EWE_CONFIG)
        self.logger.info(f"  {len(ee_urls)} articles Éwé trouvés")

        if not ee_urls:
            self.logger.warning("Aucun article trouvé. Vérifiez la connexion.")
            return records

        checkpoint = self.load_checkpoint() or {"processed_urls": []}
        processed = set(checkpoint.get("processed_urls", []))

        for ee_url in tqdm(ee_urls, desc="Articles JW.org"):
            if ee_url in processed:
                continue

            if consecutive_errors >= max_consecutive_errors:
                self.logger.warning(
                    f"Arrêt après {max_consecutive_errors} erreurs consécutives. "
                    f"Sauvegarde de {len(records)} paires collectées."
                )
                break

            try:
                fr_url = self._convert_url_to_other_lang(
                    ee_url, JW_EWE_CONFIG, JW_FR_CONFIG
                )

                ee_html = self._fetch(ee_url).text
                fr_html = self._fetch(fr_url).text

                ee_paras = self._extract_paragraphs(ee_html)
                fr_paras = self._extract_paragraphs(fr_html)

                # Alignement par position (les articles ont la même structure)
                for ee_p, fr_p in zip(ee_paras, fr_paras):
                    records.append({
                        "fr": fr_p,
                        "ee": ee_p,
                        "source": "jw_article",
                    })

                processed.add(ee_url)
                consecutive_errors = 0  # Reset si succès

                # Checkpoint toutes les 20 pages
                if len(processed) % 20 == 0:
                    self.save_checkpoint({"processed_urls": list(processed)})
                    if records:
                        self.save_records(records, "articles.jsonl")

            except Exception as e:
                consecutive_errors += 1
                self.logger.warning(
                    f"Erreur article ({consecutive_errors}/{max_consecutive_errors}): {e}"
                )

        self.logger.info(f"Articles JW.org: {len(records)} paires de paragraphes")
        if records:
            self.save_records(records, "articles.jsonl")
        return records
