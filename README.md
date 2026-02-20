# ewe-nllb

Traduction **Francais <-> Ewe** avec NLLB-200 fine-tune.

Modele [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) de Meta, fine-tune avec LoRA sur ~40 000 paires de phrases paralleles francais-ewe.

## Installation

```bash
pip install ewe-nllb
```

Pour la synthese vocale (TTS) :

```bash
pip install ewe-nllb[tts]
```

## Utilisation

### Python

```python
from ewe_nllb import translate

# Francais -> Ewe
translate("Bonjour, comment allez-vous ?")
# => "Ŋdi, aleke wòle?"

# Ewe -> Francais
translate("Ŋdi", src="ee", tgt="fr")
# => "Bonjour"
```

### Utilisation avancee

```python
from ewe_nllb import EweTranslator

t = EweTranslator()

# Traduction simple
t.translate("Le Ghana est un beau pays.", src="fr", tgt="ee")

# Traduction par lot
t.translate_batch(["Bonjour", "Merci", "Au revoir"], src="fr", tgt="ee")
```

### Synthese vocale (TTS)

```python
from ewe_nllb import EweTTS

tts = EweTTS()
waveform = tts.synthesize("Ŋdi, aleke wòle?")
tts.save_audio(waveform, "output.wav")
```

### Ligne de commande

```bash
# Francais -> Ewe
ewe-translate "Bonjour, comment allez-vous ?"

# Ewe -> Francais
ewe-translate "Ŋdi" --src ee --tgt fr

# Avec synthese vocale
ewe-translate "Bonjour" --tts --output audio.wav
```

## Donnees d'entrainement

Le modele a ete fine-tune sur des donnees provenant de :

| Source | Paires | Type |
|--------|--------|------|
| [Masakhane MAFAND](https://github.com/masakhane-io/masakhane-mt) | ~5 000 | Corpus academique |
| eBible.org | ~31 000 | Versets bibliques |
| JW.org | ~4 750 | Bible + articles |

Dataset complet : [cnss-ewe-project/library](https://huggingface.co/datasets/cnss-ewe-project/library)

## Modele

- **Base** : facebook/nllb-200-distilled-600M (600M parametres)
- **Fine-tuning** : LoRA (rank=16, alpha=32) sur les couches d'attention
- **Parametres entraines** : 2.4M / 1404M (0.2%)
- **Eval loss** : 2.02

Modele sur HuggingFace : [cnss-ewe-project/nllb-ewe-finetuned](https://huggingface.co/cnss-ewe-project/nllb-ewe-finetuned)

## Langues supportees

| Code | Langue |
|------|--------|
| `fr` | Francais |
| `ee` | Ewe (Eʋegbe) |

L'ewe est parle par environ 7 millions de personnes au Ghana et au Togo.

## Licence

MIT
