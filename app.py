import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD FILES ----------------

model = pickle.load(open("model .pkl", "rb"))
vectorizer = pickle.load(open("vectorizer .pkl", "rb"))

# ---------------- SIDEBAR ----------------

st.sidebar.title("📌 About")

st.sidebar.info("""
 Sentiment Analysis Dashboard

Model:
- TF-IDF
- SVM Classifier

Features:
✅ Sentiment Prediction
✅ Probability Scores
✅ Dashboard Visualization
✅ Download Report
""")

# ---------------- TITLE ----------------

st.title("🧠 Sentiment Analysis Dashboard")

st.write("""
Analyze customer reviews using NLP and Machine Learning.
""")

# ---------------- INPUT ----------------

review = st.text_area(
    "✍ Enter Customer Review",
    height=150
)

# ---------------- BUTTON ----------------

if st.button("🔍 Analyze"):

    if review.strip() == "":
        st.warning("Please enter review")

    else:

        # Vectorize
        review_vector = vectorizer.transform([review])

        # Prediction
        prediction = model.predict(review_vector)

        # Probability
        probabilities = model.predict_proba(review_vector)

        result = prediction[0]

        # ---------------- RESULT ----------------

        st.subheader("Prediction Result")

        if result == "Positive":
            st.success("😊 Positive Sentiment")

        elif result == "Negative":
            st.error("😞 Negative Sentiment")

        else:
            st.warning("😐 Neutral Sentiment")

        # ---------------- PROBABILITY SCORES ----------------

        st.subheader("📊 Probability Scores")

        labels = model.classes_

        prob_df = pd.DataFrame({
            "Sentiment": labels,
            "Probability": probabilities[0]
        })

        st.dataframe(prob_df)

        # ---------------- BAR CHART ----------------

        st.subheader("📈 Sentiment Dashboard")

        fig, ax = plt.subplots()

        ax.bar(labels, probabilities[0])

        ax.set_ylabel("Probability")
        ax.set_xlabel("Sentiment")

        st.pyplot(fig)

        # ---------------- DOWNLOAD REPORT ----------------

        st.subheader("⬇ Download Prediction Report")

        report = f"""
Sentiment Analysis Report

Review:
{review}

Predicted Sentiment:
{result}

Probability Scores:
"""

        for label, prob in zip(labels, probabilities[0]):
            report += f"\n{label}: {prob:.4f}"

        st.download_button(
            label="Download Report",
            data=report,
            file_name="sentiment_report.txt",
            mime="text/plain"
        )

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption("Built with Streamlit & Machine Learning")
