"""Upload du dataset vers Hugging Face Hub."""

import logging

from datasets import DatasetDict
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


class DatasetUploader:
    """Upload les datasets parallèle et monolingue vers Hugging Face Hub."""

    def __init__(self, repo_name: str, token: str):
        self.repo_name = repo_name
        self.token = token
        self.api = HfApi(token=token)

    def upload(
        self,
        parallel_dataset: DatasetDict,
        monolingual_dataset: DatasetDict,
        dataset_card_content: str,
    ):
        """Upload les deux configurations et la carte du dataset."""

        # Créer le repo si nécessaire
        logger.info(f"Création/vérification du repo: {self.repo_name}")
        self.api.create_repo(
            repo_id=self.repo_name,
            repo_type="dataset",
            exist_ok=True,
        )

        # Upload des données parallèles (config "translation")
        logger.info("Upload des données parallèles (translation)...")
        parallel_dataset.push_to_hub(
            self.repo_name,
            config_name="translation",
            token=self.token,
        )

        # Upload des données monolingues (config "monolingual")
        logger.info("Upload des données monolingues (monolingual)...")
        monolingual_dataset.push_to_hub(
            self.repo_name,
            config_name="monolingual",
            token=self.token,
        )

        # Upload de la carte du dataset (README.md)
        logger.info("Upload de la carte du dataset...")
        self.api.upload_file(
            path_or_fileobj=dataset_card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=self.repo_name,
            repo_type="dataset",
            token=self.token,
        )

        logger.info(
            f"Upload terminé ! Dataset disponible sur: "
            f"https://huggingface.co/datasets/{self.repo_name}"
        )
