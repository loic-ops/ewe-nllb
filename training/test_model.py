"""
Test du modèle NLLB fine-tuné pour la traduction Français ↔ Éwé.

Usage:
    python training/test_model.py
    python training/test_model.py --direction ee2fr
    python training/test_model.py --text "Bonjour tout le monde"
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "nllb-ewe-finetuned" / "merged"

SRC_LANG = "fra_Latn"
TGT_LANG = "ewe_Latn"
MAX_LENGTH = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """Charge le modèle et le tokenizer."""
    logger.info(f"Chargement du modèle depuis {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = model.to(device)
    model.eval()
    logger.info(f"Modèle chargé sur {device}")
    return model, tokenizer, device


def translate(model, tokenizer, device, text, src_lang, tgt_lang):
    """Traduit un texte."""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=MAX_LENGTH,
            num_beams=5,
        )

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Test du modèle NLLB fr↔ee")
    parser.add_argument("--text", type=str, help="Texte à traduire")
    parser.add_argument("--direction", type=str, default="fr2ee",
                        choices=["fr2ee", "ee2fr"], help="Direction de traduction")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR), help="Chemin du modèle")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    # Si un texte est fourni, traduire juste celui-là
    if args.text:
        if args.direction == "fr2ee":
            result = translate(model, tokenizer, device, args.text, SRC_LANG, TGT_LANG)
            print(f"FR: {args.text}")
            print(f"EE: {result}")
        else:
            result = translate(model, tokenizer, device, args.text, TGT_LANG, SRC_LANG)
            print(f"EE: {args.text}")
            print(f"FR: {result}")
        return

    # === Tests automatiques ===
    print("\n" + "=" * 70)
    print("  TEST DU MODÈLE NLLB FINE-TUNÉ (Français ↔ Éwé)")
    print("=" * 70)

    # --- Français → Éwé ---
    print("\n--- FRANÇAIS → ÉWÉ ---\n")
    fr_sentences = [
        # Salutations
        "Bonjour, comment allez-vous ?",
        "Bonsoir, je suis content de vous voir.",
        "Merci beaucoup.",
        # Phrases simples
        "Je suis un étudiant.",
        "Il fait beau aujourd'hui.",
        "L'eau est très importante pour la vie.",
        "Les enfants jouent dans la cour.",
        # Géographie / Culture
        "Le Ghana est un beau pays.",
        "Lomé est la capitale du Togo.",
        "L'Afrique est un grand continent.",
        # Phrases plus complexes
        "Je vais au marché pour acheter des légumes.",
        "Ma mère cuisine très bien.",
        "Nous devons travailler ensemble pour réussir.",
        "L'éducation est la clé du développement.",
        # Phrases religieuses (présentes dans les données d'entraînement)
        "Dieu est amour.",
        "Au commencement, Dieu créa les cieux et la terre.",
        # Phrases du quotidien
        "Quelle heure est-il ?",
        "Je voudrais un verre d'eau, s'il vous plaît.",
        "Comment tu t'appelles ?",
        "Où est l'hôpital ?",
    ]

    for sentence in fr_sentences:
        result = translate(model, tokenizer, device, sentence, SRC_LANG, TGT_LANG)
        print(f"  FR: {sentence}")
        print(f"  EE: {result}")
        print()

    # --- Éwé → Français ---
    print("\n--- ÉWÉ → FRANÇAIS ---\n")
    ee_sentences = [
        "Ŋdi, aleke wòle?",
        "Akpe na wo.",
        "Nye ŋkɔe nye Kofi.",
        "Ghana nye dukɔ nyui aɖe.",
        "Mawu nye lɔlɔ̃.",
        "Deviwo le tefã me.",
        "Nuka wòle nuwɔwɔm?",
        "Eʋegbe nye gbe nyui aɖe.",
    ]

    for sentence in ee_sentences:
        result = translate(model, tokenizer, device, sentence, TGT_LANG, SRC_LANG)
        print(f"  EE: {sentence}")
        print(f"  FR: {result}")
        print()

    print("=" * 70)
    print("  TESTS TERMINÉS")
    print("=" * 70)


if __name__ == "__main__":
    main()
