import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="Text Similarity Analyzer", layout="wide")
st.title("Text Similarity Analyzer")
st.caption("Model: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Transformers · No preprocessing · No training")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def cosine_sim(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

# ── Input ────────────────────────────────────────────────────────────────────
st.subheader("Enter Sentences / Words")
st.info("Enter one sentence or word per line (minimum 2 lines).")

default_text = "I love machine learning\nArtificial intelligence is amazing\nDeep learning is a subset of AI\nI enjoy playing football\nThe weather is sunny today"
user_input = st.text_area("Input Text", value=default_text, height=160)
reference = st.text_input("Reference Sentence", value="I love machine learning")

if st.button("Analyze", type="primary"):
    lines = [l.strip() for l in user_input.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        st.error("Please enter at least 2 lines.")
        st.stop()

    with st.spinner("Computing embeddings..."):
        all_texts = lines + [reference]
        all_emb = get_embeddings(all_texts)
        line_emb = all_emb[:len(lines)]
        ref_emb  = all_emb[-1]

        scores = [cosine_sim(ref_emb, line_emb[i]) for i in range(len(lines))]
        score_df = pd.DataFrame({"Sentence": lines, "Similarity Score": scores}).sort_values("Similarity Score", ascending=False)

        pairwise = np.array([[cosine_sim(line_emb[i], line_emb[j]) for j in range(len(lines))] for i in range(len(lines))])

    # ── Paul's Standards Scoring ──────────────────────────────────────────────
    avg_len   = np.mean([len(l.split()) for l in lines])
    spread    = float(np.std(scores))
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))
    off_diag  = float((pairwise.sum() - np.trace(pairwise)) / max(len(lines)*(len(lines)-1), 1))

    paul = {
        "Clarity":      min(100, int((avg_len / 15) * 100)),
        "Accuracy":     100,
        "Precision":    min(100, int(spread * 500)),
        "Relevance":    min(100, int(off_diag * 100)),
        "Logic":        min(100, int(max_score * 100)),
        "Significance": min(100, int(max_score * 100)),
        "Fairness":     min(100, max(20, int((1 - min_score) * 100))),
    }

    # ── Graphs ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Graph 1 · Similarity Scores (Bar Chart)**")
        fig_bar = px.bar(
            score_df, x="Similarity Score", y="Sentence", orientation="h",
            color="Similarity Score", color_continuous_scale="Blues", range_x=[0,1],
            text=score_df["Similarity Score"].apply(lambda x: f"{x:.3f}"),
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=350)
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("**Graph 2 · Pairwise Similarity (Heatmap)**")
        short = [l[:25]+"…" if len(l)>25 else l for l in lines]
        fig_heat = go.Figure(go.Heatmap(
            z=pairwise, x=short, y=short,
            colorscale="RdBu", zmin=0, zmax=1,
            text=np.round(pairwise, 2), texttemplate="%{text}",
        ))
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("**Graph 3 · Paul's Critical Thinking Standards Score**")
    standards = list(paul.keys())
    values    = list(paul.values())
    fig_paul = go.Figure(go.Bar(
        x=standards, y=values,
        marker_color=["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD","#E0AC69"],
        text=[f"{v}%" for v in values], textposition="outside",
    ))
    fig_paul.update_layout(yaxis=dict(range=[0,115], title="Score (%)"), xaxis_title="Paul's Standard", height=380)
    st.plotly_chart(fig_paul, use_container_width=True)

    # ── Critical Thinking Notes ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Paul's Critical Thinking Standards — Analysis Notes")

    top_s = score_df.iloc[0]["Sentence"]; top_v = score_df.iloc[0]["Similarity Score"]
    bot_s = score_df.iloc[-1]["Sentence"]; bot_v = score_df.iloc[-1]["Similarity Score"]

    notes = {
        "Clarity":      f"Input had {len(lines)} sentences with avg length {avg_len:.1f} words. Reference: *\"{reference}\"*.",
        "Accuracy":     f"Model: `{MODEL_NAME}`. No preprocessing applied. Scores are raw cosine similarity values.",
        "Precision":    f"Exact scores shown for all inputs. Highest: **{top_v:.4f}**, Lowest: **{bot_v:.4f}**, Spread: **{top_v-bot_v:.4f}**.",
        "Relevance":    f"All three graphs directly support results — bar chart ranks sentences, heatmap shows pairwise relations, standards chart evaluates analysis quality.",
        "Logic":        f"Top result **\"{top_s}\"** scored **{top_v:.4f}** — high cosine similarity confirms semantic closeness to reference in embedding space.",
        "Significance": f"Most important result: **\"{top_s}\"** with score **{top_v:.4f}**. This sentence is the closest semantic match to the reference.",
        "Fairness":     f"Limitation: model may miss domain-specific or culturally nuanced meaning. Lowest match **\"{bot_s}\"** ({bot_v:.4f}) may still be relevant but scored low due to surface dissimilarity.",
    }

    for std, note in notes.items():
        v = paul[std]
        color = "#2ecc71" if v >= 70 else "#f39c12" if v >= 40 else "#e74c3c"
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:8px 14px;margin-bottom:10px;background:#f9f9f9;border-radius:4px;">'
            f'<b>{std}</b> &nbsp;<span style="color:{color};font-weight:bold;">{v}%</span><br>{note}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Raw Similarity Scores")
    st.dataframe(score_df.reset_index(drop=True), use_container_width=True)
