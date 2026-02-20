"""Téléchargement et parsing du corpus Masakhane MAFAND fr-ewe."""

from pathlib import Path
from tqdm import tqdm

from scrapers.base_scraper import BaseScraper
from config import MASAKHANE_TSV_FILES


class MasakhaneLoader(BaseScraper):
    """
    Charge le corpus parallèle Masakhane MAFAND (fr-ewe).
    Source : https://github.com/masakhane-io/lafand-mt
    ~23 000 paires de phrases fr-ee.
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir, delay=0.5)

    def collect(self) -> list[dict]:
        self.logger.info("Téléchargement du corpus Masakhane MAFAND fr-ewe...")
        records = []

        for split_name, url in tqdm(MASAKHANE_TSV_FILES.items(), desc="Masakhane"):
            try:
                response = self._fetch(url)
                lines = response.text.strip().split("\n")

                for i, line in enumerate(lines):
                    # Sauter le header (première ligne: "fr\tewe")
                    if i == 0 and line.strip().lower().startswith("fr"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        fr_text = parts[0].strip()
                        ee_text = parts[1].strip()
                        if fr_text and ee_text:
                            records.append({
                                "fr": fr_text,
                                "ee": ee_text,
                                "source": "masakhane_mafand",
                                "split_origin": split_name,
                            })

                self.logger.info(
                    f"  {split_name}: {len(lines)} lignes chargées"
                )
            except Exception as e:
                self.logger.error(f"Erreur sur {split_name}: {e}")
                # Fallback : essayer via la librairie datasets
                records.extend(self._fallback_hf_datasets(split_name))

        self.logger.info(f"Masakhane total: {len(records)} paires parallèles")
        self.save_records(records)
        return records

    def _fallback_hf_datasets(self, split_name: str) -> list[dict]:
        """Fallback: charger via la librairie HuggingFace datasets."""
        try:
            from datasets import load_dataset

            self.logger.info(f"Fallback HF datasets pour {split_name}...")
            ds = load_dataset("masakhane/mafand", "fr-ewe", split=split_name)
            records = []
            for item in ds:
                translation = item.get("translation", {})
                fr_text = translation.get("fr", "").strip()
                ee_text = translation.get("ee", "").strip()
                if fr_text and ee_text:
                    records.append({
                        "fr": fr_text,
                        "ee": ee_text,
                        "source": "masakhane_mafand_hf",
                        "split_origin": split_name,
                    })
            return records
        except Exception as e:
            self.logger.error(f"Fallback HF datasets échoué: {e}")
            return []
