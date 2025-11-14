# Analisador de Sentimentos em Português com Emojis 🙂 😕 😡 😢

Projeto de **análise de sentimentos em português** usando emojis como rótulos.
O modelo recebe uma frase (com ou sem emojis) e retorna a probabilidade de ela
estar associada a quatro emoções:

- 🙂 Feliz  
- 😕 Confuso  
- 😡 Bravo  
- 😢 Triste  

A ideia é lidar com linguagem informal em PT-BR (gírias, xingamentos, abreviações)
e mostrar claramente qual emoção é mais provável para cada texto.

---

## 1. Visão geral

Este repositório contém:

- pipeline de preparação de dados a partir de planilhas Excel com textos e emojis;
- treinamento de modelos de machine learning clássicos (usando `scikit-learn`);
- scripts de inferência para testar frases;
- uma interface construída em Python (Streamlit) para digitar frases e visualizar as
  probabilidades por emoção.

---

## 2. Estrutura do projeto

```text
.
├── app.py                  # Interface (Streamlit)
├── inference.py            # Funções de inferência / predição
├── train.py                # Script de treino do modelo clássico
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # dados brutos (ex: treino.xlsx) - não versionados
│   ├── external/           # dados externos (ex: dataset_sentimentos_pt_200k.xlsx)
│   └── processed/
│       └── treino_clean.parquet   # dataset limpo (gerado no treino)
└── models/
    └── classic/
        ├── vectorizer.pkl         # vetorizador treinado (TF-IDF, etc.)
        └── model.pkl              # modelo de classificação treinado 


