"""Synthese vocale pour l'Ewe avec facebook/mms-tts-ewe."""

import logging

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

logger = logging.getLogger(__name__)

TTS_MODEL_NAME = "facebook/mms-tts-ewe"


class EweTTS:
    """Synthese vocale pour l'Ewe.

    Utilise le modele facebook/mms-tts-ewe (VITS) pour convertir
    du texte ewe en audio.

    Usage:
        >>> from ewe_nllb import EweTTS
        >>> tts = EweTTS()
        >>> waveform = tts.synthesize("Ŋdi")
        >>> tts.save_audio(waveform, "output.wav")
    """

    def __init__(self):
        logger.info(f"Chargement du modele TTS: {TTS_MODEL_NAME}")
        self.model = VitsModel.from_pretrained(TTS_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
        self.model.eval()
        self.sample_rate = self.model.config.sampling_rate
        logger.info(f"TTS charge (sample rate: {self.sample_rate} Hz)")

    def synthesize(self, text: str) -> np.ndarray:
        """Genere l'audio a partir du texte en Ewe.

        Args:
            text: Texte en Ewe a synthetiser.

        Returns:
            Array numpy contenant le signal audio.
        """
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs)

        return output.waveform[0].cpu().numpy()

    def save_audio(self, waveform: np.ndarray, output_path: str):
        """Sauvegarde l'audio en fichier WAV.

        Args:
            waveform: Signal audio (numpy array).
            output_path: Chemin du fichier WAV de sortie.
        """
        import scipy.io.wavfile as wavfile

        wavfile.write(output_path, self.sample_rate, waveform)
        logger.info(f"Audio sauvegarde: {output_path}")
