"""Loader OPUS pour corpus parallèles fr-ee."""

import xml.etree.ElementTree as ET
from pathlib import Path

from scrapers.base_scraper import BaseScraper


# Corpus OPUS susceptibles de contenir de l'Éwé
OPUS_CORPORA_TO_TRY = [
    "Tatoeba",
    "wikimedia",
    "bible-uedin",
    "GNOME",
    "Ubuntu",
    "QED",
    "GlobalVoices",
    "TED2020",
]


class OPUSLoader(BaseScraper):
    """
    Tente de télécharger des corpus parallèles fr-ee depuis OPUS.
    Note : JW300 a été retiré d'OPUS pour raisons de copyright.
    Le rendement peut être faible ou nul.
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, delay=1.0)

    def _try_opustools(self) -> list[dict]:
        """Essaie d'utiliser opustools pour télécharger les données."""
        records = []
        try:
            from opustools import OpusGet
        except ImportError:
            self.logger.warning(
                "opustools non installé. Installer avec: pip install opustools"
            )
            return records

        for corpus_name in OPUS_CORPORA_TO_TRY:
            try:
                self.logger.info(f"  Essai corpus OPUS: {corpus_name}")
                download_dir = str(self.output_dir / corpus_name)

                og = OpusGet(
                    source="fr",
                    target="ee",
                    directory=corpus_name,
                    preprocess="xml",
                    download_dir=download_dir,
                )
                og.get_files()

                # Parser les fichiers téléchargés
                corpus_records = self._parse_downloaded_files(Path(download_dir))
                if corpus_records:
                    self.logger.info(
                        f"  {corpus_name}: {len(corpus_records)} paires trouvées"
                    )
                    records.extend(corpus_records)
                else:
                    self.logger.info(f"  {corpus_name}: aucune donnée fr-ee")

            except Exception as e:
                self.logger.debug(f"  {corpus_name} non disponible pour fr-ee: {e}")

        return records

    def _parse_downloaded_files(self, download_dir: Path) -> list[dict]:
        """Parse les fichiers TMX ou alignés téléchargés par opustools."""
        records = []

        # Chercher les fichiers TMX
        for tmx_file in download_dir.rglob("*.tmx"):
            records.extend(self._parse_tmx(tmx_file))

        # Chercher les fichiers texte alignés
        fr_files = sorted(download_dir.rglob("*.fr"))
        ee_files = sorted(download_dir.rglob("*.ee"))

        for fr_file, ee_file in zip(fr_files, ee_files):
            fr_lines = fr_file.read_text(encoding="utf-8").strip().split("\n")
            ee_lines = ee_file.read_text(encoding="utf-8").strip().split("\n")

            for fr_line, ee_line in zip(fr_lines, ee_lines):
                fr_text = fr_line.strip()
                ee_text = ee_line.strip()
                if fr_text and ee_text:
                    records.append({
                        "fr": fr_text,
                        "ee": ee_text,
                        "source": f"opus_{download_dir.name}",
                    })

        return records

    def _parse_tmx(self, tmx_path: Path) -> list[dict]:
        """Parse un fichier TMX (Translation Memory eXchange)."""
        records = []
        try:
            tree = ET.parse(tmx_path)
            root = tree.getroot()

            for tu in root.iter("tu"):
                fr_text = ""
                ee_text = ""

                for tuv in tu.iter("tuv"):
                    lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                    lang = lang.lower()
                    seg = tuv.find("seg")

                    if seg is not None and seg.text:
                        if lang in ("fr", "fra"):
                            fr_text = seg.text.strip()
                        elif lang in ("ee", "ewe"):
                            ee_text = seg.text.strip()

                if fr_text and ee_text:
                    records.append({
                        "fr": fr_text,
                        "ee": ee_text,
                        "source": f"opus_{tmx_path.stem}",
                    })

        except ET.ParseError as e:
            self.logger.warning(f"Erreur parsing TMX {tmx_path}: {e}")

        return records

    def collect(self) -> list[dict]:
        self.logger.info("Chargement corpus OPUS (fr-ee)...")
        records = self._try_opustools()

        if not records:
            self.logger.info(
                "Aucune donnée OPUS trouvée pour fr-ee. "
                "Ceci est attendu (JW300 retiré, Éwé peu représenté)."
            )

        self.logger.info(f"OPUS total: {len(records)} paires parallèles")
        if records:
            self.save_records(records)
        return records
