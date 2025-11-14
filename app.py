"""
UI (Streamlit) para análise de sentimentos em 4 emojis.
Requisitos de UI/UX atendidos:
- Layout centralizado e responsivo
- Campo de texto com foco automático ao carregar
- Botão "Analisar" e "Usar exemplos" (6 frases)
- Exibe 4 cards (um por emoji) com probabilidade (0–100%)
- Card vencedor com escala ~1.15, borda destacada e tooltip "Mais provável"
- Mostra JSON de saída (scores, label_top, score_top)
- Se score_top < 0.45, exibe aviso "Baixa confiança"
- Loga tempo de inferência no console (feito em inference.py)

Acessibilidade:
- Contraste AA via CSS
- aria-labels nos cards e elementos interativos
"""

from __future__ import annotations

import json
import time

import streamlit as st

from inference import predict_proba

EMOJIS = ["🙂", "😕", "😡", "😢"]
CARD_TITLES = {
    "🙂": "Feliz",
    "😕": "Confuso",
    "😡": "Bravo",
    "😢": "Triste",
}

EXEMPLOS = [
    "Que notícia maravilhosa! Estou radiante de alegria! 😍",
    "Muito obrigado pela ajuda, deu tudo certo! 🙂",
    "Não tenho certeza se isso faz sentido… estou em dúvida.",
    "Isso é um absurdo! Cansei dessa situação! 😡",
    "Estou realmente irritado com esse atendimento.",
    "Fiquei bem chateado com a notícia, foi decepcionante. 😢",
]

st.set_page_config(page_title="Análise de Sentimentos (🙂 😕 😡 😢)", page_icon="💬", layout="centered")


st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
        text-align: center;
    }
    .cards {
        display: grid;
        grid-template-columns: repeat(2, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 1rem;
    }
    .card {
        border: 2px solid rgba(120, 120, 120, 0.35);
        border-radius: 16px;
        padding: 14px 16px;
        background: #111418;
        color: #f2f2f2;
        box-shadow: 0 2px 12px rgba(0,0,0,0.20);
        transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }
    .card h3 {
        margin: 0 0 6px 0;
        font-size: 1.1rem;
    }
    .card .emoji {
        font-size: 1.6rem;
        margin-right: 6px;
    }
    .card .prob {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .card.winner {
        transform: scale(1.15);
        border-color: #27ae60; /* contraste AA com fundo escuro */
        box-shadow: 0 6px 18px rgba(39,174,96,0.35);
    }
    .jsonbox {
        width: 100%;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.95rem;
        background: #0b0e11;
        color: #e6e6e6;
        border: 1px solid #2b2f36;
        border-radius: 12px;
        padding: 12px 14px;
        overflow-x: auto;
    }
    .tooltip {
        position: relative;
        cursor: default;
    }
    .tooltip:hover::after {
        content: attr(data-title);
        position: absolute;
        top: -36px;
        left: 50%;
        transform: translateX(-50%);
        padding: 6px 8px;
        background: #1f6f3f;
        color: #fff;
        border-radius: 6px;
        font-size: 0.80rem;
        white-space: nowrap;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='centered'><h1 aria-label='Título do aplicativo'>Análise de Sentimentos (PT-BR)</h1>"
    "<p>Insira um texto e veja as probabilidades para: 🙂, 😕, 😡, 😢</p></div>",
    unsafe_allow_html=True,
)


st.markdown(
    """
    <script>
    window.addEventListener('load', function() {
        const ta = window.parent.document.querySelector('textarea[aria-label="Entrada de texto"]');
        if (ta) { ta.focus(); }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

texto = st.text_area(
    "Digite o texto para análise:",
    key="input_text",
    height=130,
    help="Campo de texto em PT-BR (aceita acentos e emoji).",
    placeholder="Ex.: Estou muito feliz com o resultado! 🎉",
)

col_b1, col_b2 = st.columns([1, 1])
with col_b1:
    analisar = st.button("Analisar", type="primary")
with col_b2:
    usar_exemplos = st.button("Usar exemplos")

if usar_exemplos:
    st.markdown("#### Exemplos rápidos")
    
    ex_cols = st.columns(3)
    for i, frase in enumerate(EXEMPLOS):
        with ex_cols[i % 3]:
            if st.button(f"Ex{i+1}", key=f"ex_{i}", help=frase, use_container_width=True):
                st.session_state["input_text"] = frase
                st.experimental_rerun()

if analisar:
    if not texto.strip():
        st.warning("Digite um texto antes de analisar.", icon="⚠️")
    else:
        
        result = predict_proba(texto)

        scores = result["scores"]
        label_top = result["label_top"]
        score_top = float(result["score_top"])

        
        st.markdown("<div class='cards' role='list' aria-label='Probabilidades por classe'>", unsafe_allow_html=True)
        for emj in EMOJIS:
            prob = scores.get(emj, 0.0)
            perc = round(prob * 100, 2)
            is_winner = (emj == label_top)
            classes = "card winner tooltip" if is_winner else "card"
            tooltip = " data-title='Mais provável' " if is_winner else ""
            aria = f"aria-label='Card {CARD_TITLES[emj]} {emj} probabilidade {perc} por cento'"

            st.markdown(
                f"""
                <div class="{classes}" {tooltip} {aria} role="listitem" title="{CARD_TITLES[emj]}">
                    <h3><span class="emoji">{emj}</span>{CARD_TITLES[emj]}</h3>
                    <div class="prob">{perc:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Aviso de baixa confiança
        if score_top < 0.45:
            st.info("**Baixa confiança**: a previsão principal tem score < 0.45.", icon="ℹ️")

        # JSON de saída
        pretty = {
            "scores": {k: round(float(v), 6) for k, v in scores.items()},  # softmax ~1.0
            "label_top": label_top,
            "score_top": round(score_top, 6),
        }
        st.markdown("#### Saída (JSON)")
        st.markdown(f"<div class='jsonbox' aria-label='Bloco JSON'><pre>{json.dumps(pretty, ensure_ascii=False, indent=2)}</pre></div>", unsafe_allow_html=True)
