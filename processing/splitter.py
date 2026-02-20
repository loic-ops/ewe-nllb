"""Découpage des données en splits train/validation/test."""

import logging
import random

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Découpe les enregistrements en train/validation/test."""

    def split(
        self,
        records: list[dict],
        test_size: float = 0.1,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> dict[str, list[dict]]:
        """
        Découpe les données en 3 splits.
        Par défaut : 80% train, 10% validation, 10% test.
        """
        if not records:
            return {"train": [], "validation": [], "test": []}

        # Mélanger les données
        shuffled = list(records)
        random.seed(seed)
        random.shuffle(shuffled)

        n = len(shuffled)
        test_count = int(n * test_size)
        val_count = int(n * val_size)

        test = shuffled[:test_count]
        val = shuffled[test_count : test_count + val_count]
        train = shuffled[test_count + val_count :]

        logger.info(
            f"Split: {len(train)} train, {len(val)} validation, {len(test)} test"
        )

        return {"train": train, "validation": val, "test": test}
