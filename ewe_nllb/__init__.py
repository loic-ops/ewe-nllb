"""ewe-nllb : Traduction Francais <-> Ewe avec NLLB-200 fine-tune."""

__version__ = "0.1.0"

from ewe_nllb.translator import EweTranslator
from ewe_nllb.tts import EweTTS

# Instance globale (lazy loading)
_translator = None


def translate(text: str, src: str = "fr", tgt: str = "ee", **kwargs) -> str:
    """Traduit un texte entre francais et ewe.

    Le modele est telecharge automatiquement au premier appel.

    Args:
        text: Texte a traduire.
        src: Langue source ("fr" ou "ee").
        tgt: Langue cible ("fr" ou "ee").

    Returns:
        Texte traduit.

    Examples:
        >>> from ewe_nllb import translate
        >>> translate("Bonjour")
        'Ŋdi'
        >>> translate("Ŋdi", src="ee", tgt="fr")
        'Bonjour'
    """
    global _translator
    if _translator is None:
        _translator = EweTranslator()
    return _translator.translate(text, src=src, tgt=tgt, **kwargs)


__all__ = ["translate", "EweTranslator", "EweTTS", "__version__"]
