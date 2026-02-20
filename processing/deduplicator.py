"""Dédoublonnage exact et fuzzy des enregistrements."""

import hashlib
import logging

logger = logging.getLogger(__name__)


class Deduplicator:
    """Supprime les doublons exacts et quasi-doublons."""

    def deduplicate_exact(self, records: list[dict]) -> list[dict]:
        """Supprime les doublons exacts basés sur le hash du texte normalisé."""
        seen = set()
        unique = []

        for record in records:
            # Construire une clé à partir du texte
            if "fr" in record and "ee" in record:
                key_text = record["fr"].lower().strip() + "|||" + record["ee"].lower().strip()
            elif "ee" in record:
                key_text = record["ee"].lower().strip()
            else:
                unique.append(record)
                continue

            h = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(record)

        removed = len(records) - len(unique)
        logger.info(f"Dédoublonnage exact: {removed} doublons supprimés sur {len(records)}")
        return unique

    def deduplicate_fuzzy(
        self, records: list[dict], threshold: float = 0.9
    ) -> list[dict]:
        """
        Supprime les quasi-doublons par similarité Jaccard de n-grammes.
        Utilisé après le dédoublonnage exact pour réduire la taille du set.
        """
        if len(records) > 50000:
            logger.warning(
                f"Dédoublonnage fuzzy ignoré: {len(records)} entrées "
                f"(trop coûteux en O(n²)). Utilisez MinHash pour les gros datasets."
            )
            return records

        unique = []
        unique_ngrams = []

        for record in records:
            text = self._get_text(record).lower()
            ngrams = self._char_ngrams(text, n=3)

            is_duplicate = False
            for existing_ngrams in unique_ngrams:
                similarity = self._jaccard(ngrams, existing_ngrams)
                if similarity > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(record)
                unique_ngrams.append(ngrams)

        removed = len(records) - len(unique)
        logger.info(f"Dédoublonnage fuzzy: {removed} quasi-doublons supprimés")
        return unique

    def _get_text(self, record: dict) -> str:
        if "fr" in record and "ee" in record:
            return record["fr"] + " " + record["ee"]
        return record.get("ee", record.get("text", ""))

    def _char_ngrams(self, text: str, n: int = 3) -> set[str]:
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _jaccard(self, set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
