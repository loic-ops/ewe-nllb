"""
Fine-tuning NLLB-200 pour la traduction Français ↔ Éwé avec LoRA (PEFT).

Modèle de base : facebook/nllb-200-distilled-600M
Dataset : données locales fr-ee
GPU : Apple MPS (Metal Performance Shaders) - 16 GB

LoRA ne fine-tune que ~2% des paramètres, ce qui réduit drastiquement
l'utilisation mémoire (compatible 16 GB).

Usage:
    python training/train_nllb.py
    python training/train_nllb.py --epochs 5
    python training/train_nllb.py --batch_size 2
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# === Configuration ===
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = PROJECT_ROOT / "models" / "nllb-ewe-finetuned"
DATA_DIR = PROJECT_ROOT / "data" / "final" / "parallel"

# Codes de langues NLLB pour Français et Éwé
SRC_LANG = "fra_Latn"  # Français
TGT_LANG = "ewe_Latn"  # Éwé

MAX_LENGTH = 64  # Longueur max des tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def detect_device():
    """Détecte le meilleur device disponible."""
    if torch.backends.mps.is_available():
        logger.info("Utilisation du GPU Apple MPS")
        return "mps"
    elif torch.cuda.is_available():
        logger.info(f"Utilisation du GPU CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        logger.info("Utilisation du CPU (lent)")
        return "cpu"


def load_local_dataset():
    """Charge le dataset depuis les fichiers locaux."""
    splits = {}
    for split_name in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        if not path.exists():
            logger.warning(f"Fichier manquant: {path}")
            continue

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Convertir en format Dataset
        splits[split_name] = Dataset.from_dict({
            "fr": [r["fr"] for r in records],
            "ee": [r["ee"] for r in records],
        })
        logger.info(f"  {split_name}: {len(records)} paires")

    return DatasetDict(splits)


def preprocess_function(examples, tokenizer):
    """Tokenize les paires fr → ee."""
    tokenizer.src_lang = SRC_LANG

    model_inputs = tokenizer(
        examples["fr"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )

    tokenizer.src_lang = TGT_LANG
    labels = tokenizer(
        examples["ee"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning NLLB-200 fr↔ee (LoRA)")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (plus élevé pour LoRA)")
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME, help="Modèle de base"
    )
    args = parser.parse_args()

    device = detect_device()

    # === 1. Charger le tokenizer et le modèle ===
    logger.info(f"Chargement du modèle {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # === 2. Appliquer LoRA ===
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,                    # Rang des matrices LoRA
        lora_alpha=32,           # Facteur de scaling
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Attention layers
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Modèle LoRA: {trainable/1e6:.1f}M paramètres entraînables "
        f"/ {total/1e6:.0f}M total ({100*trainable/total:.1f}%)"
    )

    # === 3. Charger le dataset ===
    logger.info("Chargement du dataset...")
    dataset = load_local_dataset()

    # === 4. Tokenizer les données ===
    logger.info("Tokenization des données...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["fr", "ee"],
        desc="Tokenization",
    )

    # === 5. Data collator ===
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        max_length=MAX_LENGTH,
    )

    # === 6. Arguments d'entraînement ===
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=500,
        optim="adafactor",
        # Évaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Optimisation mémoire pour Apple MPS (16 GB)
        fp16=False,
        gradient_accumulation_steps=8,  # Batch effectif = batch_size * 8
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        # Logging
        logging_steps=50,
        logging_dir=str(PROJECT_ROOT / "logs" / "training"),
        report_to="none",
        # Sauvegarde
        save_total_limit=2,
    )

    # === 7. Trainer ===
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # === 8. Entraînement ===
    logger.info("=" * 60)
    logger.info("DÉBUT DE L'ENTRAÎNEMENT (LoRA)")
    logger.info(f"  Modèle: {args.model}")
    logger.info(f"  LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    logger.info(f"  Paramètres entraînables: {trainable/1e6:.1f}M / {total/1e6:.0f}M")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size} x 8 (gradient accumulation) = {args.batch_size * 8}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Train: {len(tokenized_dataset['train'])} exemples")
    logger.info(f"  Validation: {len(tokenized_dataset['validation'])} exemples")
    logger.info("=" * 60)

    trainer.train()

    # === 9. Sauvegarder le modèle final ===
    logger.info(f"Sauvegarde du modèle dans {OUTPUT_DIR}...")
    # Sauvegarder les adapteurs LoRA
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Aussi fusionner et sauvegarder le modèle complet pour l'inférence
    merged_dir = OUTPUT_DIR / "merged"
    logger.info(f"Fusion LoRA → modèle complet dans {merged_dir}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    logger.info("Entraînement terminé !")
    logger.info(f"Adapteurs LoRA: {OUTPUT_DIR}")
    logger.info(f"Modèle fusionné: {merged_dir}")

    # === 10. Test rapide ===
    logger.info("\n=== TEST RAPIDE ===")
    test_sentences = [
        "Bonjour, comment allez-vous ?",
        "Je suis un étudiant.",
        "Le Ghana est un beau pays.",
    ]

    merged_model.eval()
    if device == "mps":
        merged_model = merged_model.to("mps")

    tokenizer.src_lang = SRC_LANG
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        if device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            generated = merged_model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                max_new_tokens=MAX_LENGTH,
            )

        translation = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"  FR: {sentence}")
        logger.info(f"  EE: {translation}")
        logger.info("")


if __name__ == "__main__":
    main()
