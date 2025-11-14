"""
Inferência para análise de sentimentos (4 emojis).
Exponibiliza a função predict_proba(text: str) -> dict no formato:

{
  "scores": {"🙂":0.12,"😕":0.08,"😡":0.25,"😢":0.55},
  "label_top":"😢",
  "score_top":0.55
}

- Carrega vectorizer.pkl e model.pkl de models/classic/
- Detecção leve de idioma: heurística para PT-BR (log de aviso se suspeitar não-PT)
- Log de tempo de inferência no console (objetivo < 300 ms em frases curtas)
"""

from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
from joblib import load

ALLOWED_LABELS = ["🙂", "😕", "😡", "😢"]
VECTORIZER_PATH = os.path.join("models", "classic", "vectorizer.pkl")
MODEL_PATH = os.path.join("models", "classic", "model.pkl")

_VECTOR = None
_MODEL = None


def _lazy_load():
    global _VECTOR, _MODEL
    if _VECTOR is None:
        if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Artefatos não encontrados. Rode 'python train.py' para treinar e salvar os modelos."
            )
        _VECTOR = load(VECTORIZER_PATH)
        _MODEL = load(MODEL_PATH)


def _likely_pt_br(text: str) -> bool:
    """
    Heurística simples e leve (sem dependências externas):
    - presença de caracteres típicos (ã, õ, ç) OU
    - presença de palavras muito comuns do PT-BR.
    """
    t = text.lower()
    specials = any(ch in t for ch in ("ã", "õ", "ç"))
    common_words = sum(w in t for w in [" de ", " que ", " não ", " é ", " para ", " com ", " estou ", " muito "])
    return specials or (common_words >= 1)


def predict_proba(text: str) -> Dict:
    """
    Executa inferência e retorna dicionário com scores por emoji, label_top e score_top.
    """
    _lazy_load()

    t0 = time.perf_counter()

    
    if not _likely_pt_br(text):
        print("[AVISO] Texto possivelmente não está em PT-BR. Prosseguindo mesmo assim.")

    X = _VECTOR.transform([text])
    proba = _MODEL.predict_proba(X)[0]  

    
    
    ordered_labels = list(_MODEL.classes_)
    
    scores = {lbl: float(proba[i]) for i, lbl in enumerate(ordered_labels)}

    for emj in ALLOWED_LABELS:
        scores.setdefault(emj, 0.0)

    label_top = max(scores.items(), key=lambda kv: kv[1])[0]
    score_top = scores[label_top]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[INFERÊNCIA] Tempo: {elapsed_ms:.2f} ms | Top: {label_top} ({score_top:.4f})")

    return {
        "scores": scores,
        "label_top": label_top,
        "score_top": float(score_top),
    }


if __name__ == "__main__":
    
    example = "Estou muito feliz com o resultado! 🎉"
    print(predict_proba(example))
