"""Nettoyage de texte avec support spécifique pour l'Éwé."""

import re
import unicodedata

from bs4 import BeautifulSoup


class TextCleaner:
    """
    Nettoie le texte brut en préservant les caractères spécifiques à l'Éwé :
    Ɖ/ɖ, Ɛ/ɛ, Ƒ/ƒ, Ŋ/ŋ, Ɔ/ɔ et les diacritiques tonaux.
    """

    def clean(self, text: str) -> str:
        # 1. Supprimer les balises HTML résiduelles
        text = BeautifulSoup(text, "html.parser").get_text()

        # 2. Normalisation Unicode NFC
        # Critique pour l'Éwé : compose les caractères de base + diacritiques
        text = unicodedata.normalize("NFC", text)

        # 3. Supprimer les caractères de contrôle (garder newlines et tabs)
        text = "".join(
            c
            for c in text
            if not unicodedata.category(c).startswith("C") or c in ("\n", "\t")
        )

        # 4. Normaliser les espaces
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 5. Supprimer URLs et emails
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # 6. Normaliser les guillemets
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")

        return text.strip()

    def clean_record(self, record: dict) -> dict:
        """Nettoie tous les champs texte d'un enregistrement."""
        cleaned = dict(record)
        for key in ("fr", "ee", "text"):
            if key in cleaned and isinstance(cleaned[key], str):
                cleaned[key] = self.clean(cleaned[key])
        return cleaned
