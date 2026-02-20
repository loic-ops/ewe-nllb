"""Construction des objets HuggingFace Dataset à partir des données traitées."""

import logging

from datasets import Dataset, DatasetDict, Features, Value, Translation

logger = logging.getLogger(__name__)


class EweDatasetBuilder:
    """Construit les DatasetDict HuggingFace pour les données parallèles et monolingues."""

    def build_parallel_dataset(self, splits: dict[str, list[dict]]) -> DatasetDict:
        """
        Construit un DatasetDict pour les données parallèles fr-ee.
        Format : {"translation": {"fr": "...", "ee": "..."}, "source": "..."}
        """
        features = Features({
            "translation": Translation(languages=["fr", "ee"]),
            "source": Value("string"),
        })

        dataset_dict = {}
        for split_name, records in splits.items():
            hf_records = []
            for r in records:
                hf_records.append({
                    "translation": {"fr": r["fr"], "ee": r["ee"]},
                    "source": r.get("source", "unknown"),
                })
            dataset_dict[split_name] = Dataset.from_list(hf_records, features=features)
            logger.info(f"  Parallel {split_name}: {len(hf_records)} entrées")

        return DatasetDict(dataset_dict)

    def build_monolingual_dataset(self, splits: dict[str, list[dict]]) -> DatasetDict:
        """
        Construit un DatasetDict pour les données monolingues Éwé.
        Format : {"text": "...", "source": "..."}
        """
        features = Features({
            "text": Value("string"),
            "source": Value("string"),
        })

        dataset_dict = {}
        for split_name, records in splits.items():
            hf_records = []
            for r in records:
                text = r.get("ee", r.get("text", ""))
                hf_records.append({
                    "text": text,
                    "source": r.get("source", "unknown"),
                })
            dataset_dict[split_name] = Dataset.from_list(hf_records, features=features)
            logger.info(f"  Monolingual {split_name}: {len(hf_records)} entrées")

        return DatasetDict(dataset_dict)
