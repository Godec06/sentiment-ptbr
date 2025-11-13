# project/train.py
"""
Treino de classificador de sentimentos (PT-BR) em 4 classes/emoji:
🙂 (feliz), 😕 (confuso), 😡 (bravo), 😢 (triste)

- Lê SUA base em data/raw/treino1.xlsx (aba 'dados', colunas text + label_emoji)
- Lê base extra em data/external/dataset_sentimentos_pt_200k.xlsx
  (colunas 'texto' + 'emoji' ou 'text' + 'label_emoji')
- (Opcional) lê patch em data/external/patch_sentimentos_4x500.xlsx
- Junta tudo em memória e treina TF-IDF + SGDClassifier por épocas
- Salva parquet limpo em data/processed/treino_clean.parquet
- Salva vectorizer.pkl e model.pkl em models/classic/

Dependências: pandas, scikit-learn, numpy, joblib, openpyxl, pyarrow
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Caminhos e constantes
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

USER_TRAIN_PATH = BASE_DIR / "data" / "raw" / "treino.xlsx"
EXTRA_TRAIN_PATH = BASE_DIR / "data" / "external" / "dataset_sentimentos_pt_200k.xlsx"
PATCH_PATH = BASE_DIR / "data" / "external" / "patch_sentimentos_4x500.xlsx"

PROCESSED_PARQUET_PATH = BASE_DIR / "data" / "processed" / "treino_clean.parquet"

ALLOWED_LABELS = ["🙂", "😕", "😡", "😢"]

VECTORIZER_PATH = BASE_DIR / "models" / "classic" / "vectorizer.pkl"
MODEL_PATH = BASE_DIR / "models" / "classic" / "model.pkl"


# ------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------

def ensure_dirs() -> None:
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models" / "classic").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models" / "finetune").mkdir(parents=True, exist_ok=True)


def validate_excel(df: pd.DataFrame, keep_duplicates: bool) -> pd.DataFrame:
    """
    Garante colunas text + label_emoji e labels válidos.
    Remove vazios e, opcionalmente, duplicados (texto + emoji).
    """
    required_cols = {"text", "label_emoji"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Colunas obrigatórias ausentes. Esperado {required_cols}, "
            f"encontrei: {list(df.columns)}"
        )

    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label_emoji"] = df["label_emoji"].astype(str).str.strip()

    df = df[(df["text"] != "") & (df["label_emoji"] != "")]
    df = df.dropna(subset=["text", "label_emoji"])

    invalid = sorted(list(set(df["label_emoji"].unique()) - set(ALLOWED_LABELS)))
    if invalid:
        raise ValueError(
            f"Labels inválidos em 'label_emoji': {invalid}. "
            f"Permitidos: {ALLOWED_LABELS}"
        )

    if not keep_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["text", "label_emoji"])
        removed = before - len(df)
        if removed > 0:
            print(f"[INFO] Removidas {removed} linhas duplicadas (texto + emoji).")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza mínima de texto."""
    df = df.copy()
    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


# ------------------------------------------------------------
# Carregamento das bases
# ------------------------------------------------------------

def _load_user_train(keep_duplicates: bool) -> pd.DataFrame:
    if not USER_TRAIN_PATH.exists():
        print(f"[LOAD] Base principal NÃO encontrada em {USER_TRAIN_PATH}, pulando.")
        return pd.DataFrame(columns=["text", "label_emoji"])

    print(f"[LOAD] Lendo base principal: {USER_TRAIN_PATH}")
    df = pd.read_excel(USER_TRAIN_PATH, sheet_name="dados", engine="openpyxl")
    df = validate_excel(df, keep_duplicates=keep_duplicates)
    return df[["text", "label_emoji"]].copy()


def _load_extra_200k(keep_duplicates: bool) -> pd.DataFrame:
    if not EXTRA_TRAIN_PATH.exists():
        print(f"[LOAD] Base extra 200k NÃO encontrada em {EXTRA_TRAIN_PATH}, pulando.")
        return pd.DataFrame(columns=["text", "label_emoji"])

    print(f"[LOAD] Lendo base extra 200k: {EXTRA_TRAIN_PATH}")
    df_raw = pd.read_excel(EXTRA_TRAIN_PATH, sheet_name=0, engine="openpyxl")

    if {"texto", "emoji"}.issubset(df_raw.columns):
        df = pd.DataFrame()
        df["text"] = df_raw["texto"].astype(str)
        df["label_emoji"] = df_raw["emoji"].astype(str)
    elif {"text", "label_emoji"}.issubset(df_raw.columns):
        df = df_raw[["text", "label_emoji"]].copy()
    else:
        raise ValueError(
            f"[200k] Não encontrei formato esperado. "
            f"Esperado ('texto','emoji') ou ('text','label_emoji'). "
            f"Colunas atuais: {list(df_raw.columns)}"
        )

    df = validate_excel(df, keep_duplicates=keep_duplicates)
    return df[["text", "label_emoji"]].copy()


def _load_patch(keep_duplicates: bool) -> pd.DataFrame:
    if not PATCH_PATH.exists():
        return pd.DataFrame(columns=["text", "label_emoji"])

    print(f"[LOAD] Lendo patch extra: {PATCH_PATH}")
    df = pd.read_excel(PATCH_PATH, sheet_name="dados", engine="openpyxl")
    df = validate_excel(df, keep_duplicates=keep_duplicates)
    return df[["text", "label_emoji"]].copy()


def load_training_data(keep_duplicates: bool) -> pd.DataFrame:
    """
    Junta: SUA base (treino1) + base 200k + patch (se existirem).
    Se só uma existir, usa só aquela.
    """
    dfs: List[pd.DataFrame] = []

    df_user = _load_user_train(keep_duplicates=keep_duplicates)
    if not df_user.empty:
        dfs.append(df_user)

    df_extra = _load_extra_200k(keep_duplicates=keep_duplicates)
    if not df_extra.empty:
        dfs.append(df_extra)

    df_patch = _load_patch(keep_duplicates=keep_duplicates)
    if not df_patch.empty:
        dfs.append(df_patch)

    if not dfs:
        raise RuntimeError(
            "Nenhuma base de treino encontrada.\n"
            f"Verifique se {USER_TRAIN_PATH} e/ou {EXTRA_TRAIN_PATH} existem."
        )

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[LOAD] Total de linhas após juntar tudo: {len(df_all)}")
    print("[LOAD] Distribuição de emojis:")
    print(df_all["label_emoji"].value_counts())
    return df_all


# ------------------------------------------------------------
# Avaliação
# ------------------------------------------------------------

def evaluate_model(vectorizer: TfidfVectorizer, clf: SGDClassifier,
                   X_val_texts: List[str], y_val: List[str]) -> None:
    X_val = vectorizer.transform(X_val_texts)
    y_pred = clf.predict(X_val)

    f1 = f1_score(y_val, y_pred, average="macro")
    print(f"[VAL] F1(macro): {f1:.4f}")
    print("[VAL] Relatório por classe:")
    print(classification_report(y_val, y_pred, digits=4))
    print("[VAL] Matriz de confusão:")
    print(confusion_matrix(y_val, y_pred, labels=ALLOWED_LABELS))


# ------------------------------------------------------------
# Treino
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20,
                        help="Nº de passagens completas pelos dados de treino.")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Tamanho do mini-batch para partial_fit.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Embaralhar os exemplos a cada época.")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Não remover linhas duplicadas (texto+emoji).")
    args = parser.parse_args()

    ensure_dirs()

    print("[INFO] Carregando bases de treino...")
    df = load_training_data(keep_duplicates=args.keep_duplicates)

    print("[INFO] Pré-processando textos...")
    df = preprocess(df)

    print(f"[INFO] Salvando parquet limpo em {PROCESSED_PARQUET_PATH}")
    df.to_parquet(PROCESSED_PARQUET_PATH, index=False)

    # Split 80/20
    print("[INFO] Split estratificado 80/20...")
    X_texts = df["text"].tolist()
    y = df["label_emoji"].tolist()

    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        X_texts, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[INFO] Tamanho treino: {len(X_train_texts)} | val: {len(X_val_texts)}")

    # Vetorizador
    token_pattern = r"(?u)\b\w+\b|[^\w\s]"
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=token_pattern,
    )

    print("[INFO] Ajustando TF-IDF no conjunto de treino...")
    X_train = vectorizer.fit_transform(X_train_texts)
    X_val = vectorizer.transform(X_val_texts)

    # Classificador incremental
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    classes = np.array(ALLOWED_LABELS)
    n = X_train.shape[0]
    bs = max(1, int(args.batch_size))
    indices = np.arange(n)

    print("[INFO] Iniciando treino incremental por épocas...")
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        if args.shuffle:
            np.random.shuffle(indices)

        seen = 0
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch_idx = indices[start:end]

            X_batch = X_train[batch_idx]
            y_batch = np.array(y_train)[batch_idx]

            if ep == 1 and start == 0:
                clf.partial_fit(X_batch, y_batch, classes=classes)
            else:
                clf.partial_fit(X_batch, y_batch)

            seen += (end - start)

        y_pred_val = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred_val, average="macro")
        print(f"[Época {ep:02d}] vistas: {seen} | F1(macro) val: {f1:.4f}")

    train_ms = (time.perf_counter() - t0) * 1000
    print(f"[INFO] Treino concluído em {train_ms:.1f} ms")

    print("[INFO] Avaliação final no conjunto de validação:")
    evaluate_model(vectorizer, clf, X_val_texts, y_val)

    print(f"[INFO] Salvando artefatos em {VECTORIZER_PATH} e {MODEL_PATH}")
    dump(vectorizer, VECTORIZER_PATH)
    dump(clf, MODEL_PATH)

    print("\n[OK] Treinamento finalizado com sucesso.")


if __name__ == "__main__":
    main()
