"""Traducteur Français <-> Ewe basé sur NLLB-200 fine-tuné."""

import logging

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Modèle fine-tuné sur HuggingFace Hub
HF_MODEL_REPO = "cnss-ewe-project/nllb-ewe-finetuned"
BASE_MODEL = "facebook/nllb-200-distilled-600M"

LANG_CODES = {
    "fr": "fra_Latn",
    "ee": "ewe_Latn",
}

MAX_LENGTH = 128


def _detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class EweTranslator:
    """Traducteur Français <-> Ewe.

    Utilise le modèle NLLB-200 fine-tuné pour la paire fr-ee.
    Le modèle est téléchargé automatiquement depuis HuggingFace Hub
    au premier usage.

    Usage:
        >>> from ewe_nllb import EweTranslator
        >>> t = EweTranslator()
        >>> t.translate("Bonjour")
        'Ŋdi'
    """

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Chemin local vers le modèle. Si None, télécharge
                        depuis HuggingFace Hub.
        """
        self.device = _detect_device()

        if model_path:
            model_name = model_path
        else:
            try:
                model_name = snapshot_download(HF_MODEL_REPO)
                logger.info(f"Modèle téléchargé depuis {HF_MODEL_REPO}")
            except Exception:
                logger.warning(
                    f"Impossible de télécharger {HF_MODEL_REPO}, "
                    f"utilisation du modèle de base {BASE_MODEL}"
                )
                model_name = BASE_MODEL

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        logger.info(f"Modèle chargé sur {self.device}")

    def translate(
        self,
        text: str,
        src: str = "fr",
        tgt: str = "ee",
        num_beams: int = 5,
        max_length: int = MAX_LENGTH,
    ) -> str:
        """Traduit un texte.

        Args:
            text: Texte source.
            src: Langue source ("fr" ou "ee").
            tgt: Langue cible ("fr" ou "ee").
            num_beams: Nombre de beams pour la recherche.
            max_length: Longueur maximale de la sortie.

        Returns:
            Texte traduit.
        """
        src_code = LANG_CODES[src]
        tgt_code = LANG_CODES[tgt]

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=max_length, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                max_new_tokens=max_length,
                num_beams=num_beams,
            )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def translate_batch(
        self,
        texts: list[str],
        src: str = "fr",
        tgt: str = "ee",
        num_beams: int = 5,
    ) -> list[str]:
        """Traduit une liste de textes.

        Args:
            texts: Liste de textes source.
            src: Langue source ("fr" ou "ee").
            tgt: Langue cible ("fr" ou "ee").
            num_beams: Nombre de beams.

        Returns:
            Liste de textes traduits.
        """
        src_code = LANG_CODES[src]
        tgt_code = LANG_CODES[tgt]

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                max_new_tokens=MAX_LENGTH,
                num_beams=num_beams,
            )

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
