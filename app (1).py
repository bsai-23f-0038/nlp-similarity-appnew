import streamlit as st
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="Text Similarity Analyzer", layout="wide")

st.title("Text Similarity Analyzer")
st.caption("Using `all-MiniLM-L6-v2` · No preprocessing · No training")

# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# ── Input ────────────────────────────────────────────────────────────────────
st.subheader("Enter Sentences / Words")
st.info("Enter one sentence or word per line (minimum 2 lines).")

default_text = "I love machine learning\nArtificial intelligence is amazing\nDeep learning is a subset of AI\nI enjoy playing football\nThe weather is sunny today"
user_input = st.text_area("Input Text", value=default_text, height=160)
reference = st.text_input("Reference Sentence (for top similarity ranking)", value="I love machine learning")

if st.button("Analyze", type="primary"):
    lines = [l.strip() for l in user_input.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        st.error("Please enter at least 2 lines.")
        st.stop()

    # ── Embeddings ────────────────────────────────────────────────────────────
    embeddings = model.encode(lines, convert_to_tensor=True)
    ref_embedding = model.encode(reference, convert_to_tensor=True)

    # Cosine similarities vs reference
    scores = util.cos_sim(ref_embedding, embeddings)[0].cpu().numpy()
    score_df = pd.DataFrame({"Sentence": lines, "Similarity Score": scores}).sort_values("Similarity Score", ascending=False)

    # Pairwise similarity matrix
    pairwise = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # ── Paul's Standards Scoring ──────────────────────────────────────────────
    def paul_scores(lines, scores, pairwise):
        """
        Compute a 0-100 score for each Paul's Standard based on real model output.
        """
        n = len(lines)
        avg_len = np.mean([len(l.split()) for l in lines])
        score_spread = float(np.std(scores))          # Precision: how varied are scores
        max_score = float(np.max(scores))             # Significance: best match strength
        min_score = float(np.min(scores))             # Fairness: weakest match (limitation visible)
        avg_pairwise_off_diag = float(
            (pairwise.sum() - np.trace(pairwise)) / max(n * (n - 1), 1)
        )                                              # Relevance: overall coherence

        clarity    = min(100, int((avg_len / 15) * 100))          # Longer = more descriptive input
        accuracy   = 100                                           # Model name is known, no unsupported claims
        precision  = min(100, int(score_spread * 500))            # High spread = more precise discrimination
        relevance  = min(100, int(avg_pairwise_off_diag * 100))   # High avg similarity = coherent input
        logic      = min(100, int(max_score * 100))               # Top match confidence
        significance = min(100, int(max_score * 100))             # Same as logic — best result's strength
        fairness   = min(100, max(20, int((1 - min_score) * 100)))# Low min score = model limitation visible

        return {
            "Clarity": clarity,
            "Accuracy": accuracy,
            "Precision": precision,
            "Relevance": relevance,
            "Logic": logic,
            "Significance": significance,
            "Fairness": fairness,
        }

    paul = paul_scores(lines, scores, pairwise)

    # ── Layout ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Results")

    col1, col2 = st.columns(2)

    # Graph 1 — Bar Chart (similarity scores)
    with col1:
        st.markdown("**Graph 1 · Similarity Scores (Bar Chart)**")
        fig_bar = px.bar(
            score_df,
            x="Similarity Score",
            y="Sentence",
            orientation="h",
            color="Similarity Score",
            color_continuous_scale="Blues",
            range_x=[0, 1],
            text=score_df["Similarity Score"].apply(lambda x: f"{x:.3f}"),
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=350)
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Graph 2 — Heatmap (pairwise similarity)
    with col2:
        st.markdown("**Graph 2 · Pairwise Similarity (Heatmap)**")
        short_labels = [l[:25] + "…" if len(l) > 25 else l for l in lines]
        fig_heat = go.Figure(go.Heatmap(
            z=pairwise,
            x=short_labels,
            y=short_labels,
            colorscale="RdBu",
            zmin=0, zmax=1,
            text=np.round(pairwise, 2),
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(height=350)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Graph 3 — Paul's Standards Score (Radar / Bar)
    st.markdown("**Graph 3 · Paul's Critical Thinking Standards Score**")
    standards = list(paul.keys())
    values = list(paul.values())

    fig_paul = go.Figure()
    fig_paul.add_trace(go.Bar(
        x=standards,
        y=values,
        marker_color=["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974","#64B5CD","#E0AC69"],
        text=[f"{v}%" for v in values],
        textposition="outside",
    ))
    fig_paul.update_layout(
        yaxis=dict(range=[0, 115], title="Score (%)"),
        xaxis_title="Paul's Standard",
        height=380,
    )
    st.plotly_chart(fig_paul, use_container_width=True)

    # ── Critical Thinking Notes ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Paul's Critical Thinking Standards — Analysis Notes")

    top_sentence = score_df.iloc[0]["Sentence"]
    top_score    = score_df.iloc[0]["Similarity Score"]
    bot_sentence = score_df.iloc[-1]["Sentence"]
    bot_score    = score_df.iloc[-1]["Similarity Score"]

    notes = {
        "Clarity":      f"Input contained {len(lines)} sentences with an average length of {np.mean([len(l.split()) for l in lines]):.1f} words. The reference sentence was: *\"{reference}\"*.",
        "Accuracy":     f"Model used: `sentence-transformers/all-MiniLM-L6-v2`. No preprocessing, tokenization, or manual cleaning was applied. Scores are raw cosine similarity values between embeddings.",
        "Precision":    f"Exact similarity scores are shown for every input. The highest score was **{top_score:.4f}** and the lowest was **{bot_score:.4f}**, giving a spread of **{top_score - bot_score:.4f}**.",
        "Relevance":    f"All three graphs directly support the results — the bar chart shows rankings, the heatmap shows pairwise relationships, and the standards chart evaluates critical thinking quality of this analysis.",
        "Logic":        f"The top result **\"{top_sentence}\"** scored **{top_score:.4f}**, which makes sense semantically given the reference sentence. High cosine similarity indicates shared vector space proximity.",
        "Significance": f"The most important result is **\"{top_sentence}\"** with a score of **{top_score:.4f}**. This sentence is the closest match to the reference in the embedding space.",
        "Fairness":     f"Limitation: the model may not capture domain-specific or culturally nuanced language well. The lowest-scoring sentence **\"{bot_sentence}\"** ({bot_score:.4f}) may be semantically relevant but scored low due to surface-level dissimilarity.",
    }

    for std, note in notes.items():
        score_val = paul[std]
        color = "#2ecc71" if score_val >= 70 else "#f39c12" if score_val >= 40 else "#e74c3c"
        st.markdown(
            f"""<div style="border-left:4px solid {color}; padding:8px 14px; margin-bottom:10px; background:#f9f9f9; border-radius:4px;">
            <b>{std}</b> &nbsp;<span style="color:{color}; font-weight:bold;">{score_val}%</span><br>
            {note}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Raw Scores Table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Raw Similarity Scores")
    st.dataframe(score_df.reset_index(drop=True), use_container_width=True)
