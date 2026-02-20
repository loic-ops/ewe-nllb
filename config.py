"""Configuration centrale pour le projet Bibliothèque Universelle de l'Éwé."""

from pathlib import Path

# === Chemins ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
LOGS_DIR = PROJECT_ROOT / "logs"

# === HuggingFace ===
HF_DATASET_NAME = "cnss-ewe-project/library"

# === Masakhane MAFAND ===
MASAKHANE_BASE_URL = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data"
MASAKHANE_TSV_FILES = {
    "train": f"{MASAKHANE_BASE_URL}/tsv_files/fr-ewe/train.tsv",
    "dev": f"{MASAKHANE_BASE_URL}/tsv_files/fr-ewe/dev.tsv",
    "test": f"{MASAKHANE_BASE_URL}/tsv_files/fr-ewe/test.tsv",
}

# === JW.org ===
JW_EWE_CONFIG = {
    "iface": "ee",
    "region": "r114",
    "lang_pref": "lp-ew",
}
JW_FR_CONFIG = {
    "iface": "fr",
    "region": "r30",
    "lang_pref": "lp-f",
}

# === Wikipedia ===
WIKI_API_URL = "https://ee.wikipedia.org/w/api.php"

# === eBible ===
EBIBLE_CORPUS_URL = "https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus"
EBIBLE_VREF_URL = "https://raw.githubusercontent.com/BibleNLP/ebible/main/metadata/vref.txt"
EBIBLE_EWE_FILE = "ewe-ewe.txt"
EBIBLE_FRENCH_FILE = "fra-fraLSG.txt"

# === Scraping ===
REQUEST_DELAY = 2.0  # secondes entre les requêtes
REQUEST_TIMEOUT = 30  # secondes
MAX_RETRIES = 3
USER_AGENT = "EweUniversalLibrary/1.0 (Academic Research; Ewe Language Dataset)"

# === Filtres qualité ===
MIN_SENTENCE_LENGTH = 5  # caractères
MAX_SENTENCE_LENGTH = 1000  # caractères
MAX_LENGTH_RATIO = 3.0  # ratio max longueur entre fr et ee
