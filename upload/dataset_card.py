"""Génération de la carte du dataset (README.md) pour Hugging Face."""

from datasets import DatasetDict


def generate_card(
    parallel_ds: DatasetDict,
    monolingual_ds: DatasetDict,
) -> str:
    """Génère le contenu de la carte du dataset avec les statistiques réelles."""

    # Calculer les statistiques
    parallel_total = sum(len(ds) for ds in parallel_ds.values())
    mono_total = sum(len(ds) for ds in monolingual_ds.values())

    parallel_stats = {name: len(ds) for name, ds in parallel_ds.items()}
    mono_stats = {name: len(ds) for name, ds in monolingual_ds.items()}

    # Déterminer la catégorie de taille
    total = parallel_total + mono_total
    if total < 1000:
        size_cat = "n<1K"
    elif total < 10000:
        size_cat = "1K<n<10K"
    elif total < 100000:
        size_cat = "10K<n<100K"
    elif total < 1000000:
        size_cat = "100K<n<1M"
    else:
        size_cat = "1M<n<10M"

    card = f"""---
language:
  - ee
  - fr
license: cc-by-sa-4.0
multilinguality: bilingual
size_categories:
  - {size_cat}
task_categories:
  - translation
  - text-generation
task_ids:
  - language-modeling
tags:
  - ewe
  - eʋegbe
  - low-resource
  - african-languages
  - parallel-corpus
  - monolingual-corpus
  - ghana
  - togo
pretty_name: "Bibliothèque Universelle de l'Éwé"
configs:
  - config_name: translation
    data_files:
      - split: train
        path: translation/train-*
      - split: validation
        path: translation/validation-*
      - split: test
        path: translation/test-*
  - config_name: monolingual
    data_files:
      - split: train
        path: monolingual/train-*
      - split: validation
        path: monolingual/validation-*
      - split: test
        path: monolingual/test-*
---

# Bibliothèque Universelle de l'Éwé (Universal Library of Ewe)

## Description

Un dataset complet pour la langue Éwé (Eʋegbe), combinant des données de
traduction parallèles Français-Éwé et du texte monolingue Éwé. Conçu pour
entraîner des modèles de langue capables de comprendre, générer et traduire
l'Éwé.

L'Éwé est une langue Niger-Congo parlée par environ 5 millions de personnes
au Ghana et au Togo.

## Configurations

### `translation` - Données parallèles Français-Éwé

| Split | Entrées |
|-------|---------|
| train | {parallel_stats.get('train', 0):,} |
| validation | {parallel_stats.get('validation', 0):,} |
| test | {parallel_stats.get('test', 0):,} |
| **Total** | **{parallel_total:,}** |

**Format :**
```json
{{
  "translation": {{"fr": "Bonjour, comment allez-vous ?", "ee": "Ŋdi, aleke wòle?"}},
  "source": "masakhane_mafand"
}}
```

### `monolingual` - Texte monolingue Éwé

| Split | Entrées |
|-------|---------|
| train | {mono_stats.get('train', 0):,} |
| validation | {mono_stats.get('validation', 0):,} |
| test | {mono_stats.get('test', 0):,} |
| **Total** | **{mono_total:,}** |

**Format :**
```json
{{
  "text": "Eʋegbe nye gbe si Eʋeawo ƒo le Gana kple Togo.",
  "source": "wikipedia_ee"
}}
```

## Sources

- **Masakhane MAFAND-MT** : Corpus parallèle curé par la communauté Masakhane
- **JW.org Bible** : Traduction du Nouveau Monde (NWT) en Éwé et Français
- **JW.org Articles** : Articles bilingues Watchtower et Awake!
- **eBible.org** : Texte biblique parallèle via BibleNLP
- **OPUS** : Corpus parallèles ouverts (quand disponible)
- **Wikipedia Éwé** : Articles de ee.wikipedia.org (monolingue)

## Utilisation

```python
from datasets import load_dataset

# Charger les données de traduction
translation_ds = load_dataset("cnss-ewe-project/ewe-universal-library", "translation")

# Charger les données monolingues
mono_ds = load_dataset("cnss-ewe-project/ewe-universal-library", "monolingual")

# Accéder aux données
print(translation_ds["train"][0])
# {{"translation": {{"fr": "...", "ee": "..."}}, "source": "..."}}
```

## Langues

- **Éwé (ee)** : Langue cible. Caractères spéciaux : Ɖ/ɖ, Ɛ/ɛ, Ƒ/ƒ, Ŋ/ŋ, Ɔ/ɔ
- **Français (fr)** : Langue source pour les données parallèles

## Licence

CC-BY-SA-4.0. Les sources individuelles peuvent avoir des licences différentes.

## Citation

Si vous utilisez ce dataset, merci de citer :

```bibtex
@misc{{ewe-universal-library,
  title={{Bibliothèque Universelle de l'Éwé}},
  year={{2026}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/cnss-ewe-project/ewe-universal-library}}
}}
```
"""
    return card
