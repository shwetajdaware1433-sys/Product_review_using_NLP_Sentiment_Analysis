import streamlit as st
import joblib
import re
import string
import nltk
from pathlib import Path
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------
# LOAD DATASET REVIEWS
# -------------------------------------------------------

@st.cache_data
def load_reviews():
    df = pd.read_excel("product_reviews.xlsx")   
    return df

reviews_df = load_reviews()

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis",
    layout="wide"
)

st.title("Sentiment Analysis using Multiple Models")


# -------------------------------------------------------
# DOWNLOAD NLTK DATA ONLY IF REQUIRED
# -------------------------------------------------------

@st.cache_resource
def setup_nltk():

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\\S+|www\\S+", "", text)

    text = re.sub(r"\\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------

@st.cache_resource
def load_models():


    model = joblib.load("model.pkl")

    tfidf = joblib.load( "tfidf_vectorizer.pkl")

    return model, tfidf


model, tfidf = load_models()
            

# -------------------------------------------------------
# INPUT
# -------------------------------------------------------

st.subheader("Choose a Review or Type Your Own")

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

def update_review():
    st.session_state.review_text = st.session_state.selected_review

selected_review = st.selectbox(
    "Select a review from the dataset",
    options=[""] + reviews_df["clean_review"].tolist(),
    key="selected_review",
    on_change=update_review
)

review = st.text_area(
    "Or type your own review",
    key="review_text",
    height=150,
    placeholder="Example: This mobile phone is really good..."
)
# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        confidence = model.predict_proba(vector).max()

        st.markdown("## Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Sentiment",
            str(prediction).upper()
        )

        col2.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        st.success(
            f"The review is predicted as **{prediction.upper()}** "
            f"with **{confidence:.2%}** confidence."
        )

        # Probability Distribution
        probabilities = model.predict_proba(vector)[0]

        prob_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Probability": probabilities
        })

        st.subheader("Probability Distribution")

        st.bar_chart(
            prob_df,
            x="Sentiment",
            y="Probability",
            use_container_width=True
        )

  
# -------------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------------

st.sidebar.write("""
### Model Used

- Logistic Regression
- TF-IDF Vectorizer
""")

st.sidebar.markdown("---")

st.sidebar.info(
"""
This application loads pre-trained models.
No model training occurs during startup,
making the application much faster.
"""
)