"""
Pipeline d'inférence : Traduction Français ↔ Éwé + Text-to-Speech.

Combine le modèle NLLB fine-tuné pour la traduction
et facebook/mms-tts-ewe pour la synthèse vocale en Éwé.

Usage:
    python inference/pipeline.py "Bonjour, comment allez-vous ?"
    python inference/pipeline.py "Bonjour" --tts --output audio.wav
    python inference/pipeline.py "Ŋdi, aleke wòle?" --direction ee2fr
"""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    VitsModel,
    AutoTokenizer as VitsTokenizer,
)

# === Configuration ===
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "nllb-ewe-finetuned"
BASE_MODEL = "facebook/nllb-200-distilled-600M"

# Codes de langues NLLB
LANG_CODES = {
    "fr": "fra_Latn",
    "ee": "ewe_Latn",
}

# Modèle TTS pour l'Éwé
TTS_MODEL_NAME = "facebook/mms-tts-ewe"

MAX_LENGTH = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EweTranslator:
    """Traducteur Français ↔ Éwé basé sur NLLB fine-tuné."""

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Chemin vers le modèle fine-tuné.
                        Si None, utilise le modèle de base NLLB.
        """
        self.device = self._detect_device()

        if model_path and Path(model_path).exists():
            logger.info(f"Chargement du modèle fine-tuné: {model_path}")
            self.model_name = model_path
        else:
            logger.info(f"Modèle fine-tuné non trouvé, utilisation du modèle de base: {BASE_MODEL}")
            self.model_name = BASE_MODEL

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        logger.info(f"Modèle chargé sur {self.device}")

    def _detect_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def translate(
        self,
        text: str,
        src_lang: str = "fr",
        tgt_lang: str = "ee",
        num_beams: int = 5,
        max_length: int = MAX_LENGTH,
    ) -> str:
        """
        Traduit un texte.

        Args:
            text: Texte source.
            src_lang: Langue source ("fr" ou "ee").
            tgt_lang: Langue cible ("fr" ou "ee").
            num_beams: Nombre de beams pour la recherche.
            max_length: Longueur maximale de la sortie.

        Returns:
            Texte traduit.
        """
        src_code = LANG_CODES[src_lang]
        tgt_code = LANG_CODES[tgt_lang]

        self.tokenizer.src_lang = src_code

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                max_new_tokens=max_length,
                num_beams=num_beams,
            )

        translation = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return translation

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str = "fr",
        tgt_lang: str = "ee",
        num_beams: int = 5,
    ) -> list[str]:
        """Traduit une liste de textes."""
        src_code = LANG_CODES[src_lang]
        tgt_code = LANG_CODES[tgt_lang]

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

        translations = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return translations


class EweTTS:
    """Synthèse vocale pour l'Éwé avec facebook/mms-tts-ewe."""

    def __init__(self):
        logger.info(f"Chargement du modèle TTS: {TTS_MODEL_NAME}")
        self.model = VitsModel.from_pretrained(TTS_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
        self.model.eval()
        self.sample_rate = self.model.config.sampling_rate
        logger.info(f"TTS chargé (sample rate: {self.sample_rate} Hz)")

    def synthesize(self, text: str) -> np.ndarray:
        """
        Génère l'audio à partir du texte en Éwé.

        Args:
            text: Texte en Éwé à synthétiser.

        Returns:
            Array numpy contenant le signal audio.
        """
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs)

        waveform = output.waveform[0].cpu().numpy()
        return waveform

    def save_audio(self, waveform: np.ndarray, output_path: str):
        """Sauvegarde l'audio en fichier WAV."""
        import scipy.io.wavfile as wavfile

        wavfile.write(output_path, self.sample_rate, waveform)
        logger.info(f"Audio sauvegardé: {output_path}")


class EwePipeline:
    """
    Pipeline complet : Traduction + TTS.

    Traduit du français vers l'éwé, puis génère l'audio.
    """

    def __init__(self, model_path: str = None, enable_tts: bool = True):
        self.translator = EweTranslator(model_path)
        self.tts = EweTTS() if enable_tts else None

    def translate_and_speak(
        self,
        text: str,
        src_lang: str = "fr",
        tgt_lang: str = "ee",
        output_audio: str = None,
    ) -> dict:
        """
        Traduit le texte et optionnellement génère l'audio.

        Args:
            text: Texte source.
            src_lang: Langue source.
            tgt_lang: Langue cible.
            output_audio: Chemin pour sauvegarder l'audio (None = pas d'audio).

        Returns:
            Dict avec 'source', 'translation', et optionnellement 'audio_path'.
        """
        translation = self.translator.translate(text, src_lang, tgt_lang)

        result = {
            "source": text,
            "source_lang": src_lang,
            "translation": translation,
            "target_lang": tgt_lang,
        }

        # TTS uniquement si la cible est l'éwé
        if self.tts and tgt_lang == "ee" and output_audio:
            waveform = self.tts.synthesize(translation)
            self.tts.save_audio(waveform, output_audio)
            result["audio_path"] = output_audio

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Traduction Français ↔ Éwé + TTS"
    )
    parser.add_argument("text", type=str, help="Texte à traduire")
    parser.add_argument(
        "--direction",
        type=str,
        default="fr2ee",
        choices=["fr2ee", "ee2fr"],
        help="Direction de traduction (défaut: fr2ee)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Chemin vers le modèle fine-tuné",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Activer la synthèse vocale (TTS)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Fichier audio de sortie (défaut: output.wav)",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=5,
        help="Nombre de beams (défaut: 5)",
    )
    args = parser.parse_args()

    # Déterminer les langues
    if args.direction == "fr2ee":
        src_lang, tgt_lang = "fr", "ee"
    else:
        src_lang, tgt_lang = "ee", "fr"

    # Créer le pipeline
    pipeline = EwePipeline(
        model_path=args.model,
        enable_tts=(args.tts and tgt_lang == "ee"),
    )

    # Traduire (et synthétiser)
    result = pipeline.translate_and_speak(
        text=args.text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_audio=args.output if args.tts else None,
    )

    # Afficher le résultat
    print(f"\n{'='*50}")
    print(f"  {src_lang.upper()}: {result['source']}")
    print(f"  {tgt_lang.upper()}: {result['translation']}")
    if "audio_path" in result:
        print(f"  Audio: {result['audio_path']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
