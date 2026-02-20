"""API REST pour la traduction Francais <-> Ewe."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ewe_nllb.translator import EweTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ewe-nllb API",
    description="Traduction Francais <-> Ewe avec NLLB-200 fine-tune",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques (interface web)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Chargement du modele au demarrage
translator = None


@app.on_event("startup")
async def load_model():
    global translator
    model_path = os.environ.get("MODEL_PATH")
    logger.info("Chargement du modele...")
    translator = EweTranslator(model_path=model_path)
    logger.info("Modele pret !")


class TranslateRequest(BaseModel):
    text: str
    src: str = "fr"
    tgt: str = "ee"
    num_beams: int = 5


class TranslateResponse(BaseModel):
    source: str
    translation: str
    src: str
    tgt: str


class BatchRequest(BaseModel):
    texts: list[str]
    src: str = "fr"
    tgt: str = "ee"
    num_beams: int = 5


class BatchResponse(BaseModel):
    translations: list[TranslateResponse]


@app.get("/")
async def index():
    """Page d'accueil avec l'interface web."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    """Verification de l'Etat ."""
    return {"status": "ok", "model_loaded": translator is not None}


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    """Traduit un texte."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    if req.src not in ("fr", "ee") or req.tgt not in ("fr", "ee"):
        raise HTTPException(status_code=400, detail="Langues supportees: fr, ee")

    result = translator.translate(
        req.text, src=req.src, tgt=req.tgt, num_beams=req.num_beams
    )

    return TranslateResponse(
        source=req.text,
        translation=result,
        src=req.src,
        tgt=req.tgt,
    )


@app.post("/translate/batch", response_model=BatchResponse)
async def translate_batch(req: BatchRequest):
    """Traduit plusieurs textes."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    results = translator.translate_batch(
        req.texts, src=req.src, tgt=req.tgt, num_beams=req.num_beams
    )

    translations = [
        TranslateResponse(source=text, translation=result, src=req.src, tgt=req.tgt)
        for text, result in zip(req.texts, results)
    ]

    return BatchResponse(translations=translations)
