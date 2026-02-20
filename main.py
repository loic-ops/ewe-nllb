"""
Orchestrateur principal - Bibliothèque Universelle de l'Éwé.

Pipeline complet : collecte -> traitement -> upload vers Hugging Face.

Usage:
    python main.py --collect          # Phase 1: collecte des données
    python main.py --process          # Phase 2: nettoyage et filtrage
    python main.py --upload           # Phase 3: upload vers HuggingFace
    python main.py --all              # Tout exécuter
"""

import argparse
import json
import logging
from pathlib import Path

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    FINAL_DIR,
    LOGS_DIR,
    HF_DATASET_NAME,
)


def setup_logging():
    """Configure le logging vers fichier et console."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def save_jsonl(records: list[dict], filepath: Path):
    """Sauvegarde une liste de dicts en JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info(f"Sauvegardé {len(records)} entrées dans {filepath}")


def load_jsonl(filepath: Path) -> list[dict]:
    """Charge un fichier JSONL."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logging.info(f"Chargé {len(records)} entrées depuis {filepath}")
    return records


def collect_all():
    """Phase 1 : Collecte des données depuis toutes les sources."""
    logger = logging.getLogger("collect")
    logger.info("=" * 60)
    logger.info("PHASE 1 : COLLECTE DES DONNÉES")
    logger.info("=" * 60)

    all_parallel = []
    all_monolingual = []

    # 1. Masakhane MAFAND (source la plus fiable)
    logger.info("\n--- Masakhane MAFAND ---")
    try:
        from scrapers.masakhane_loader import MasakhaneLoader

        loader = MasakhaneLoader(RAW_DIR / "masakhane")
        records = loader.collect()
        all_parallel.extend(records)
        logger.info(f"Masakhane: {len(records)} paires")
    except Exception as e:
        logger.error(f"Erreur Masakhane: {e}")

    # 2. eBible (Bible parallèle)
    logger.info("\n--- eBible ---")
    try:
        from scrapers.ebible_loader import EBibleLoader

        loader = EBibleLoader(RAW_DIR / "ebible")
        records = loader.collect()
        all_parallel.extend(records)
        logger.info(f"eBible: {len(records)} paires")
    except Exception as e:
        logger.error(f"Erreur eBible: {e}")

    # 3. JW.org Bible
    logger.info("\n--- JW.org Bible ---")
    try:
        from scrapers.jw_scraper import JWBibleScraper

        scraper = JWBibleScraper(RAW_DIR / "jw" / "bible")
        records = scraper.collect()
        all_parallel.extend(records)
        logger.info(f"JW Bible: {len(records)} paires")
    except Exception as e:
        logger.error(f"Erreur JW Bible: {e}")

    # 4. JW.org Articles
    logger.info("\n--- JW.org Articles ---")
    try:
        from scrapers.jw_scraper import JWArticleScraper

        scraper = JWArticleScraper(RAW_DIR / "jw" / "articles")
        records = scraper.collect()
        all_parallel.extend(records)
        logger.info(f"JW Articles: {len(records)} paires")
    except Exception as e:
        logger.error(f"Erreur JW Articles: {e}")

    # 5. OPUS
    logger.info("\n--- OPUS ---")
    try:
        from scrapers.opus_loader import OPUSLoader

        loader = OPUSLoader(RAW_DIR / "opus")
        records = loader.collect()
        all_parallel.extend(records)
        logger.info(f"OPUS: {len(records)} paires")
    except Exception as e:
        logger.error(f"Erreur OPUS: {e}")

    # 6. Wikipedia Éwé (monolingue)
    logger.info("\n--- Wikipedia Éwé ---")
    try:
        from scrapers.wikipedia_scraper import WikipediaEweScraper

        scraper = WikipediaEweScraper(RAW_DIR / "wikipedia")
        records = scraper.collect()
        all_monolingual.extend(records)
        logger.info(f"Wikipedia: {len(records)} phrases monolingues")
    except Exception as e:
        logger.error(f"Erreur Wikipedia: {e}")

    # Sauvegarder les données brutes
    logger.info("\n--- Sauvegarde ---")
    save_jsonl(all_parallel, PROCESSED_DIR / "parallel" / "raw_parallel.jsonl")
    save_jsonl(all_monolingual, PROCESSED_DIR / "monolingual" / "raw_monolingual.jsonl")

    logger.info(f"\nTotal parallèle: {len(all_parallel)} paires")
    logger.info(f"Total monolingue: {len(all_monolingual)} phrases")
    logger.info("Phase 1 terminée.")


def process_all():
    """Phase 2 : Nettoyage, dédoublonnage, filtrage et découpage."""
    logger = logging.getLogger("process")
    logger.info("=" * 60)
    logger.info("PHASE 2 : TRAITEMENT DES DONNÉES")
    logger.info("=" * 60)

    from processing.cleaner import TextCleaner
    from processing.deduplicator import Deduplicator
    from processing.quality_filter import QualityFilter
    from processing.splitter import DatasetSplitter

    cleaner = TextCleaner()
    dedup = Deduplicator()
    qfilter = QualityFilter()
    splitter = DatasetSplitter()

    # Traiter les données parallèles
    logger.info("\n--- Données parallèles ---")
    parallel_path = PROCESSED_DIR / "parallel" / "raw_parallel.jsonl"
    if parallel_path.exists():
        parallel = load_jsonl(parallel_path)
        logger.info(f"Brut: {len(parallel)}")

        parallel = [cleaner.clean_record(r) for r in parallel]
        parallel = dedup.deduplicate_exact(parallel)
        parallel = qfilter.filter_parallel(parallel)
        logger.info(f"Après traitement: {len(parallel)}")

        parallel_splits = splitter.split(parallel)
        for split_name, records in parallel_splits.items():
            save_jsonl(records, FINAL_DIR / "parallel" / f"{split_name}.jsonl")
    else:
        logger.warning(f"Fichier non trouvé: {parallel_path}")

    # Traiter les données monolingues
    logger.info("\n--- Données monolingues ---")
    mono_path = PROCESSED_DIR / "monolingual" / "raw_monolingual.jsonl"
    if mono_path.exists():
        mono = load_jsonl(mono_path)
        logger.info(f"Brut: {len(mono)}")

        mono = [cleaner.clean_record(r) for r in mono]
        mono = dedup.deduplicate_exact(mono)
        mono = qfilter.filter_monolingual(mono)
        logger.info(f"Après traitement: {len(mono)}")

        mono_splits = splitter.split(mono)
        for split_name, records in mono_splits.items():
            save_jsonl(records, FINAL_DIR / "monolingual" / f"{split_name}.jsonl")
    else:
        logger.warning(f"Fichier non trouvé: {mono_path}")

    logger.info("Phase 2 terminée.")


def upload_all():
    """Phase 3 : Construction des datasets HF et upload."""
    logger = logging.getLogger("upload")
    logger.info("=" * 60)
    logger.info("PHASE 3 : UPLOAD VERS HUGGING FACE")
    logger.info("=" * 60)

    from auth import authenticate
    from upload.dataset_builder import EweDatasetBuilder
    from upload.dataset_card import generate_card
    from upload.uploader import DatasetUploader

    token = authenticate()
    builder = EweDatasetBuilder()

    # Charger les splits parallèles
    parallel_splits = {}
    for name in ["train", "validation", "test"]:
        path = FINAL_DIR / "parallel" / f"{name}.jsonl"
        if path.exists():
            parallel_splits[name] = load_jsonl(path)
        else:
            logger.warning(f"Split parallèle manquant: {path}")
            parallel_splits[name] = []

    # Charger les splits monolingues
    mono_splits = {}
    for name in ["train", "validation", "test"]:
        path = FINAL_DIR / "monolingual" / f"{name}.jsonl"
        if path.exists():
            mono_splits[name] = load_jsonl(path)
        else:
            logger.warning(f"Split monolingue manquant: {path}")
            mono_splits[name] = []

    # Construire les datasets HF
    logger.info("Construction des DatasetDict...")
    parallel_ds = builder.build_parallel_dataset(parallel_splits)
    mono_ds = builder.build_monolingual_dataset(mono_splits)

    # Générer la carte
    card = generate_card(parallel_ds, mono_ds)

    # Upload
    logger.info(f"Upload vers {HF_DATASET_NAME}...")
    uploader = DatasetUploader(HF_DATASET_NAME, token)
    uploader.upload(parallel_ds, mono_ds, card)

    logger.info("Phase 3 terminée.")
    logger.info(
        f"Dataset disponible : https://huggingface.co/datasets/{HF_DATASET_NAME}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bibliothèque Universelle de l'Éwé - Pipeline de dataset"
    )
    parser.add_argument("--collect", action="store_true", help="Collecter les données")
    parser.add_argument("--process", action="store_true", help="Traiter les données")
    parser.add_argument("--upload", action="store_true", help="Upload vers HuggingFace")
    parser.add_argument("--all", action="store_true", help="Exécuter tout le pipeline")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("main")

    if not any([args.collect, args.process, args.upload, args.all]):
        parser.print_help()
        print("\nExemple: python main.py --all")
        return

    if args.all or args.collect:
        collect_all()
    if args.all or args.process:
        process_all()
    if args.all or args.upload:
        upload_all()

    logger.info("Pipeline terminé avec succès !")


if __name__ == "__main__":
    main()
