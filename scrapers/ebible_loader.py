"""Loader eBible.org pour texte parallèle Bible (fr-ee)."""

from pathlib import Path

from tqdm import tqdm

from scrapers.base_scraper import BaseScraper
from config import EBIBLE_CORPUS_URL, EBIBLE_VREF_URL, EBIBLE_EWE_FILE, EBIBLE_FRENCH_FILE


class EBibleLoader(BaseScraper):
    """
    Charge les textes bibliques parallèles depuis BibleNLP/ebible (GitHub).
    Fichiers vérifiés :
      - metadata/vref.txt (références des versets)
      - corpus/ewe-ewe.txt (Bible en Éwé)
      - corpus/fra-fraLSG.txt (Bible Louis Segond en Français)
    Alignement ligne par ligne via vref.txt.
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, delay=0.5)

    def collect(self) -> list[dict]:
        self.logger.info("Chargement corpus eBible (fr-ee)...")
        records = []

        # 1. Télécharger vref.txt (références des versets)
        try:
            self.logger.info(f"  Téléchargement vref.txt...")
            vref_response = self._fetch(EBIBLE_VREF_URL)
            vref_lines = vref_response.text.strip().split("\n")
            self.logger.info(f"  vref.txt: {len(vref_lines)} références")
        except Exception as e:
            self.logger.error(f"Impossible de télécharger vref.txt: {e}")
            return records

        # 2. Télécharger la Bible en Éwé
        try:
            ee_url = f"{EBIBLE_CORPUS_URL}/{EBIBLE_EWE_FILE}"
            self.logger.info(f"  Téléchargement {EBIBLE_EWE_FILE}...")
            ee_text = self._fetch(ee_url).text
            ee_lines = ee_text.strip().split("\n")
            self.logger.info(f"  Éwé: {len(ee_lines)} lignes")
        except Exception as e:
            self.logger.error(f"Impossible de télécharger {EBIBLE_EWE_FILE}: {e}")
            return records

        # 3. Télécharger la Bible en Français (Louis Segond)
        try:
            fr_url = f"{EBIBLE_CORPUS_URL}/{EBIBLE_FRENCH_FILE}"
            self.logger.info(f"  Téléchargement {EBIBLE_FRENCH_FILE}...")
            fr_text = self._fetch(fr_url).text
            fr_lines = fr_text.strip().split("\n")
            self.logger.info(f"  Français: {len(fr_lines)} lignes")
        except Exception as e:
            self.logger.error(f"Impossible de télécharger {EBIBLE_FRENCH_FILE}: {e}")
            return records

        # 4. Aligner par numéro de ligne (vref.txt = clé d'alignement)
        for i, ref in enumerate(tqdm(vref_lines, desc="eBible alignement")):
            ref = ref.strip()
            if not ref:
                continue

            ee_verse = ee_lines[i].strip() if i < len(ee_lines) else ""
            fr_verse = fr_lines[i].strip() if i < len(fr_lines) else ""

            if ee_verse and fr_verse:
                records.append({
                    "fr": fr_verse,
                    "ee": ee_verse,
                    "source": "ebible",
                    "ref": ref,
                })

        self.logger.info(f"eBible: {len(records)} paires de versets")
        self.save_records(records)
        return records
