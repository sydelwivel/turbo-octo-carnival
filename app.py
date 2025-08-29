import streamlit as st
import pandas as pd
import random
import json
import numpy as np
import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Ensure necessary backend modules are available
# You would need to have these files in your project directory
from analytics import (
    binomial_test,
    kl_divergence,
    js_divergence,
    earth_movers_distance,
    train_meta_classifier,
)
from personas import bias_personas
from counterfactuals import generate_counterfactuals
from ai_mock import ai_mock_score
from db import save_run_result
import data_generator
import report
import mitigation

# -----------------------------
# Load resumes (cached)
# -----------------------------
@st.cache_data
def load_resumes():
    try:
        df = pd.read_csv("assets/sample_resumes.csv")
    except FileNotFoundError:
        # Fallback to generate synthetic data if file not found
        df = data_generator.generate_synthetic(50, "assets/sample_resumes.csv")
    return df

resumes_df = load_resumes()

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Fairness Test for AI Hiring", layout="wide")
st.title("🤖 Is the AI Hiring Tool Fair? A Game for Everyone")

st.markdown("""
This tool lets you test if you can spot a hiring bot with a hidden bias. Your results will help us find and fix potential unfairness.
""")

# -----------------------------
# Session state
# -----------------------------
if "trials" not in st.session_state:
    st.session_state.trials = []

# -----------------------------
# Game: Reverse Turing Test
# -----------------------------
st.header("🎮 Spot the Biased Scorer")

if st.button("Get a New Candidate"):
    resume_idx = random.choice(resumes_df.index.tolist())
    st.session_state.current_resume_idx = resume_idx

if 'current_resume_idx' not in st.session_state:
    st.session_state.current_resume_idx = random.choice(resumes_df.index.tolist())
    
resume = resumes_df.loc[st.session_state.current_resume_idx].to_dict()

st.subheader("Candidate Profile")
st.json(resume)

# AI + persona scoring
ai_score = ai_mock_score(resume)
persona_name, persona_fn = random.choice(list(bias_personas.items()))
persona_score = persona_fn(resume)

# Shuffle order
scores = [("AI Model", ai_score), (persona_name, persona_score)]
random.shuffle(scores)

st.write("### How the Candidate Was Rated")
for idx, (src, val) in enumerate(scores):
    st.write(f"Option {idx+1}: {val:.2f}")

choice = st.radio(
    "One of these scores is from an intentionally unfair judge. The other is from an AI. Can you guess which one is the unfair judge's score?",
    [1, 2],
    index=None
)

explanation = st.text_area("Why do you think so? (optional)")

if st.button("Submit Guess"):
    if choice is None:
        st.warning("Please make a selection before submitting.")
    else:
        correct_answer = 1 if scores[choice-1][0] == persona_name else 0
        st.session_state.trials.append({
            "resume": resume,
            "ai_score": ai_score,
            "persona_score": persona_score,
            "persona": persona_name,
            "choice": choice,
            "correct": correct_answer,
            "explanation": explanation
        })
        st.success("✅ Guess submitted! Try to spot another one.")
        st.rerun()

# -----------------------------
# Sidebar: Analytics & Reports
# -----------------------------
st.sidebar.title("📊 Fairness Report")
st.sidebar.info("👉 Play the game a few times, then click **Run the Report** to see how the scores compare.")

if st.sidebar.button("Run the Report"):
    trials = st.session_state.trials
    if not trials:
        st.sidebar.warning("No guesses yet! Play the game a few times first.")
    else:
        df = pd.DataFrame(trials)
        correct = df["correct"].sum()
        n = len(df)
        
        # Use tabs for a cleaner layout
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Your Performance", "🔍 Bias Analysis", "📈 Score Distribution", "🩹 Bias Correction"])
        
        with tab1:
            st.subheader("Your Performance")
            st.write(f"You correctly identified the biased judge **{correct}/{n}** times (**{correct/n:.2%}**).")

            pval = binomial_test(correct, n, p0=0.5)
            st.write(f"The statistical significance (p-value) of your performance is **{pval:.4f}**.")
            st.write("*(A low p-value, below 0.05, suggests you're likely not guessing randomly and are good at spotting bias.)*")
            
        with tab2:
            ai_scores = df["ai_score"].values
            persona_scores = df["persona_score"].values

            st.subheader("Score Differences")
            kl = kl_divergence(ai_scores, persona_scores)
            js = js_divergence(ai_scores, persona_scores)
            emd = earth_movers_distance(ai_scores, persona_scores)

            st.write(f"**How different are the scoring patterns?**")
            st.write(f"KL Divergence: **{kl:.4f}**")
            st.write(f"JS Divergence: **{js:.4f}**")
            st.write(f"Earth Mover's Distance: **{emd:.4f}**")
            st.write("*(A higher number means the scoring patterns are more different.)*")
            
            X = df[["ai_score", "persona_score"]].values
            y = (df["persona"] != "AI Model").astype(int).values

            st.subheader("Can an AI Spot the Bias?")
            if len(np.unique(y)) > 1:
                try:
                    _, auc = train_meta_classifier(X, y)
                    st.write(f"A separate AI model can spot the bias with an AUC of **{auc:.3f}**.")
                    st.write("*(An AUC over 0.5 means the AI can find the bias, and an AUC close to 1.0 means it's very easy to spot.)*")
                except ValueError as e:
                    st.error(f"Error: {e}")
            else:
                st.info("The bias detection model needs at least one wrong guess to learn. Try to guess incorrectly at least once.")
        
        with tab3:
            st.subheader("Visualizing Score Distributions")
            st.write("The chart below shows how the AI and the biased persona scored the candidates.")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(ai_scores, label="AI Model Score", ax=ax)
            sns.kdeplot(persona_scores, label=f"Biased Persona Score", ax=ax)
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.set_title("Distribution of AI vs. Biased Persona Scores")
            ax.legend()
            st.pyplot(fig)
            
        with tab4:
            st.subheader("Bias Correction")
            st.write("Our system can automatically adjust scores to make them fairer.")
            mitigated_scores = mitigation.apply_reweighing(ai_scores, persona_scores)
            st.write("The AI's scores have been adjusted for fairness.")
            st.write(f"The new average score is: **{mitigated_scores.mean():.3f}**")
            
            st.subheader("Compliance Report")
            report_html = report.generate_report(df, ai_scores, persona_scores)
            st.download_button(
                "⬇️ Download Compliance Report (HTML)",
                data=report_html,
                file_name="fairness_report.html",
                mime="text/html"
            )
            
            save_run_result("reverse_turing_test", {"n": n}, df.to_dict(orient="records"))
            st.success("✅ All results saved.")

st.sidebar.info("👉 Submit several guesses, then click **Run the Report** to see how the scores compare.")
