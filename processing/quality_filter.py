"""Filtrage qualité des paires parallèles et textes monolingues."""

import re
import logging

from config import MIN_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH, MAX_LENGTH_RATIO

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Filtre les enregistrements de faible qualité :
    - Texte trop court/long
    - Ratio de longueur anormal (paires mal alignées)
    - Copies (même texte dans les deux langues)
    - Boilerplate (navigation, copyright)
    """

    # Caractères spécifiques à l'Éwé (utilisés pour l'heuristique de langue)
    EWE_CHARS = re.compile(r"[ɖɛƒŋɔƉƐƑŊƆ]")
    # Digrammes courants en Éwé
    EWE_DIGRAPHS = re.compile(r"(gb|kp|ny|ts|dz)", re.IGNORECASE)

    # Patterns de boilerplate à exclure
    BOILERPLATE_PATTERNS = [
        re.compile(r"^(copyright|all rights reserved|terms of use)", re.IGNORECASE),
        re.compile(r"^\d+$"),  # juste un nombre
        re.compile(r"^[.\s\-_]+$"),  # juste des points/espaces
        re.compile(r"^(menu|home|next|previous|back|search)", re.IGNORECASE),
    ]

    def filter_parallel(self, records: list[dict]) -> list[dict]:
        """Filtre une liste de paires parallèles."""
        filtered = [r for r in records if self.is_valid_parallel(r)]
        removed = len(records) - len(filtered)
        logger.info(
            f"Filtre qualité parallèle: {removed} entrées supprimées sur {len(records)}"
        )
        return filtered

    def filter_monolingual(self, records: list[dict]) -> list[dict]:
        """Filtre une liste de textes monolingues."""
        filtered = [r for r in records if self.is_valid_monolingual(r)]
        removed = len(records) - len(filtered)
        logger.info(
            f"Filtre qualité monolingue: {removed} entrées supprimées sur {len(records)}"
        )
        return filtered

    def is_valid_parallel(self, record: dict) -> bool:
        fr_text = record.get("fr", "")
        ee_text = record.get("ee", "")

        # Vérification de longueur
        if len(fr_text) < MIN_SENTENCE_LENGTH or len(ee_text) < MIN_SENTENCE_LENGTH:
            return False
        if len(fr_text) > MAX_SENTENCE_LENGTH or len(ee_text) > MAX_SENTENCE_LENGTH:
            return False

        # Ratio de longueur (détecte les paires mal alignées)
        min_len = min(len(fr_text), len(ee_text))
        max_len = max(len(fr_text), len(ee_text))
        if min_len > 0 and max_len / min_len > MAX_LENGTH_RATIO:
            return False

        # Détection de copies (même texte dans les deux langues)
        if fr_text.strip().lower() == ee_text.strip().lower():
            return False

        # Vérification boilerplate
        if self._is_boilerplate(fr_text) or self._is_boilerplate(ee_text):
            return False

        return True

    def is_valid_monolingual(self, record: dict) -> bool:
        ee_text = record.get("ee", record.get("text", ""))

        if len(ee_text) < MIN_SENTENCE_LENGTH:
            return False
        if len(ee_text) > MAX_SENTENCE_LENGTH:
            return False
        if self._is_boilerplate(ee_text):
            return False

        return True

    def _is_boilerplate(self, text: str) -> bool:
        for pattern in self.BOILERPLATE_PATTERNS:
            if pattern.match(text):
                return True
        return False
