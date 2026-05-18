import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🛍️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}

.title-style {
    color: #1f4e79;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# NLTK DOWNLOADS
# =========================
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# LOAD FILES
# =========================
BASE_DIR = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_DIR, "best_model.pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))

df = pd.read_excel(os.path.join(BASE_DIR, "dataset.xlsx"))

# Normalize columns
df.columns = df.columns.str.lower()

# =========================
# FIND REVIEW COLUMN
# =========================
possible_cols = ['cleaned_review', 'review', 'reviews', 'review_text']

review_col = None

for col in possible_cols:
    if col in df.columns:
        review_col = col
        break

if review_col is None:
    st.error("No review column found")
    st.stop()

# =========================
# NLP PREPROCESSING
# =========================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Create cleaned column if not present
if 'cleaned_review' not in df.columns:
    df['cleaned_review'] = df[review_col].apply(clean_text)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 EDA Dashboard",
        "🔍 Sentiment Prediction",
        "📈 Model Performance"
    ]
)

# =========================
# HOME PAGE
# =========================
if page == "🏠 Home":

    st.markdown(
        "<h1 class='title-style'>🛍️ Product Review Sentiment Analysis</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    This application performs sentiment analysis on customer product reviews
    using Natural Language Processing (NLP) and Machine Learning.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-box'>
        <h3>Total Reviews</h3>
        <h2>{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-box'>
        <h3>Best Model</h3>
        <h2>SVM</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-box'>
        <h3>Technique</h3>
        <h2>TF-IDF</h2>
        </div>
        """, unsafe_allow_html=True)

# =========================
# EDA PAGE
# =========================
elif page == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    # Word Count
    df['word_count'] = df['cleaned_review'].apply(
        lambda x: len(str(x).split())
    )

    st.subheader("Word Count Distribution")

    fig1, ax1 = plt.subplots(figsize=(8,4))

    ax1.hist(df['word_count'], bins=30)

    ax1.set_xlabel("Word Count")
    ax1.set_ylabel("Frequency")

    st.pyplot(fig1)

    # Review Length
    df['review_length'] = df['cleaned_review'].apply(
        lambda x: len(str(x))
    )

    st.subheader("Review Length Distribution")

    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.hist(df['review_length'], bins=30)

    ax2.set_xlabel("Review Length")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2)

    # WordCloud
    st.subheader("☁️ WordCloud")

    text = " ".join(df['cleaned_review'].dropna().astype(str))

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white'
    ).generate(text)

    fig3, ax3 = plt.subplots(figsize=(12,6))

    ax3.imshow(wordcloud, interpolation='bilinear')

    ax3.axis("off")

    st.pyplot(fig3)

# =========================
# SENTIMENT PREDICTION
# =========================
elif page == "🔍 Sentiment Prediction":

    st.title("🔍 Sentiment Analysis")

    st.write("Enter a customer review below:")

    user_input = st.text_area(
        "Customer Review",
        height=150
    )

    if st.button("Predict Sentiment"):

        if user_input.strip() == "":
            st.warning("Please enter a review")

        else:

            cleaned = clean_text(user_input)

            vector = tfidf.transform([cleaned])

            prediction = model.predict(vector)[0]

            if prediction.lower() == "positive":

                st.success(f"😊 Predicted Sentiment: {prediction}")

            elif prediction.lower() == "negative":

                st.error(f"😞 Predicted Sentiment: {prediction}")

            else:

                st.info(f"😐 Predicted Sentiment: {prediction}")

# =========================
# MODEL PERFORMANCE
# =========================
elif page == "📈 Model Performance":

    st.title("📈 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Naive Bayes",
            "SVM",
            "Random Forest",
            "KNN"
        ],
        "Accuracy": [
            0.74,
            0.70,
            0.77,
            0.72,
            0.68
        ]
    })

    st.dataframe(comparison_df)

    st.subheader("Model Accuracy Comparison")

    fig4, ax4 = plt.subplots(figsize=(8,4))

    ax4.bar(
        comparison_df["Model"],
        comparison_df["Accuracy"]
    )

    ax4.set_ylabel("Accuracy")

    plt.xticks(rotation=15)

    st.pyplot(fig4)

    st.success("""
    SVM was selected as the best model because it achieved
    the highest accuracy and performed well with TF-IDF features.
    """)
