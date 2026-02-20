"""CLI pour ewe-nllb : traduction Francais <-> Ewe."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="ewe-translate",
        description="Traduction Francais <-> Ewe avec NLLB-200 fine-tune",
    )
    parser.add_argument("text", type=str, help="Texte a traduire")
    parser.add_argument(
        "--src",
        type=str,
        default="fr",
        choices=["fr", "ee"],
        help="Langue source (defaut: fr)",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default="ee",
        choices=["fr", "ee"],
        help="Langue cible (defaut: ee)",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=5,
        help="Nombre de beams (defaut: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chemin local vers le modele (optionnel)",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Generer l'audio en ewe (TTS)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Fichier audio de sortie (defaut: output.wav)",
    )

    args = parser.parse_args()

    from ewe_nllb.translator import EweTranslator

    translator = EweTranslator(model_path=args.model)
    result = translator.translate(args.text, src=args.src, tgt=args.tgt, num_beams=args.beams)

    print(f"{args.src.upper()}: {args.text}")
    print(f"{args.tgt.upper()}: {result}")

    if args.tts and args.tgt == "ee":
        from ewe_nllb.tts import EweTTS

        tts = EweTTS()
        waveform = tts.synthesize(result)
        tts.save_audio(waveform, args.output)
        print(f"Audio: {args.output}")


if __name__ == "__main__":
    main()
