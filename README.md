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
- uma interface construída em Python (Streamlit) para digitar frases e visualizar
  as probabilidades por emoção.

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
```

## 3. Dataset


3. Dataset
Os dados são montados a partir de frases em português associadas a emojis.
Cada linha do dataset final contém, por exemplo:

texto: mensagem em português

emoji: 🙂 😕 😡 😢

label: classe de sentimento correspondente (feliz, confuso, bravo, triste)

Os arquivos principais usados no projeto são:

data/raw/treino.xlsx

data/external/dataset_sentimentos_pt_200k.xlsx

O script de treino unifica, limpa e salva uma versão consolidada em
data/processed/treino_clean.parquet.

4. Como rodar o projeto
4.1. Clonar o repositório

Copiar código

```
git clone https://github.com/Godec06/sentiment-ptbr.git
cd sentiment-ptbr
```

4.2. Criar ambiente virtual (opcional, mas recomendado)

Copiar código
```
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

4.3. Instalar dependências

Copiar código
```
pip install -r requirements.txt
```

5. Treinar o modelo
Coloque seus arquivos de dados em:

data/raw/treino.xlsx
data/external/dataset_sentimentos_pt_200k.xlsx

Depois rode:

Copiar código
```
python train.py --epochs 20 --batch-size 4096 --shuffle
```
O script vai:

carregar os datasets;

limpar e unificar os textos;
salvar data/processed/treino_clean.parquet;
treinar um modelo clássico (usando scikit-learn);
salvar o vetorizador e o modelo em:
models/classic/vectorizer.pkl
models/classic/model.pkl.

6. Fazer predições
Depois de treinar, você pode testar o modelo de duas formas.

6.1. Usando inference.py direto

Copiar código
```
python inference.py
```
O script vem com alguns exemplos de frase e imprime as probabilidades
para cada emoção no terminal.

6.2. Usando as funções de Python

Copiar código
```
python - << "EOF"
from inference import predict_proba

texto = "Eu te adoro, você é incrível! ❤️"
probs = predict_proba(texto)
print(probs)
EOF
```
A função retorna um dicionário com as probabilidades para cada classe.

7. Interface web (Streamlit)
Para abrir a interface gráfica:


Copiar código
```
streamlit run app.py
```

A interface permite:


digitar frases em PT-BR;

visualizar as probabilidades para 🙂 😕 😡 😢;

destacar a emoção mais provável;

inspecionar a saída em formato JSON.

8. Próximos passos
Algumas ideias de evolução do projeto:

ampliar e balancear ainda mais o dataset de treinamento;

testar modelos baseados em embeddings / deep learning;

adicionar métricas detalhadas (F1 por classe, matriz de confusão, etc.);

publicar a interface em um serviço online (ex.: Streamlit Cloud).

9. Autor
Pedro Godec
Análise de Dados, BI & IA • Integrações com n8n & CRMs
LinkedIn
