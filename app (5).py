import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from textblob import TextBlob

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = None
try:
    model = load_model()
except:
    st.warning("⚠️ Model file not found. Prediction will be disabled.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")
st.sidebar.info("Upload dataset + predict sentiment using ML model.")

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🤖 Predict Sentiment")

user_input = st.text_area("Enter a review:")

if st.button("Predict"):
    if model is None:
        st.error("Model not available.")
    elif user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction = model.predict([user_input])[0]

        if prediction == "Positive":
            st.success(f"😊 Sentiment: {prediction}")
        elif prediction == "Negative":
            st.error(f"😡 Sentiment: {prediction}")
        else:
            st.warning(f"😐 Sentiment: {prediction}")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])

if uploaded_file is None:
    st.info("👆 Upload dataset to see analysis")
    st.stop()

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_excel(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Check Columns
# -----------------------------
if 'body' not in df.columns or 'rating' not in df.columns:
    st.error("Dataset must contain 'body' and 'rating'")
    st.stop()

# -----------------------------
# Create Sentiment
# -----------------------------
if 'sentiment' not in df.columns:
    def get_sentiment(r):
        if r <= 2:
            return "Negative"
        elif r == 3:
            return "Neutral"
        else:
            return "Positive"
    
    df['sentiment'] = df['rating'].apply(get_sentiment)

# -----------------------------
# EDA Charts
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("⭐ Ratings")
    fig1, ax1 = plt.subplots()
    df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("😊 Sentiment")
    fig2, ax2 = plt.subplots()
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

# -----------------------------
# ABSA
# -----------------------------
st.subheader("🔥 Aspect-Based Analysis")

aspects = ['camera', 'battery', 'performance', 'display', 'wifi']

def aspect_sentiment(text, aspect):
    sentences = str(text).lower().split('.')
    
    for sentence in sentences:
        if aspect in sentence:
            polarity = TextBlob(sentence).sentiment.polarity
            
            if polarity > 0:
                return "Positive"
            elif polarity < 0:
                return "Negative"
            else:
                return "Neutral"
    return None

for aspect in aspects:
    df[aspect + '_sentiment'] = df['body'].apply(lambda x: aspect_sentiment(x, aspect))

summary_data = []

for aspect in aspects:
    counts = df[aspect + '_sentiment'].value_counts()
    
    summary_data.append({
        'Aspect': aspect.capitalize(),
        'Positive': counts.get('Positive', 0),
        'Negative': counts.get('Negative', 0),
        'Neutral': counts.get('Neutral', 0)
    })

aspect_df = pd.DataFrame(summary_data).set_index('Aspect')

fig3, ax3 = plt.subplots(figsize=(8,5))
sns.heatmap(aspect_df, annot=True, fmt='d', cmap='YlGnBu', ax=ax3)

st.pyplot(fig3)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<center>🚀 NLP Project Dashboard</center>", unsafe_allow_html=True)
