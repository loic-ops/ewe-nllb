"""Authentification Hugging Face Hub."""

import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()


def authenticate():
    """Authentifie avec Hugging Face en utilisant le token depuis .env."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN introuvable. Crée un fichier .env avec HF_TOKEN=ton_token"
        )
    login(token=token)
    print("---------------------------------------")
    print("Connexion Hugging Face établie avec succès.")
    print("---------------------------------------")
    return token


if __name__ == "__main__":
    authenticate()
