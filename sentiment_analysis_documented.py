import streamlit as st
import joblib
import re
import string
import nltk
from pathlib import Path
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------------------------------------
# LOAD DATASET REVIEWS
# -------------------------------------------------------

@st.cache_data
def load_reviews():
    df = pd.read_csv("product_reviews.csv")   
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

st.markdown(
"""
Choose a Machine Learning model from the sidebar and
predict the sentiment of a product review.
"""
)

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

    model_dir = Path("models")

    models = {

        "Logistic Regression":
            joblib.load(model_dir / "logistic_model.pkl"),

        "Naive Bayes":
            joblib.load(model_dir / "nb_model.pkl"),

        "Random Forest":
            joblib.load(model_dir / "rf_model.pkl"),

        "SVM":
            joblib.load(model_dir / "svm_model.pkl"),

        "XGBoost":
            joblib.load(model_dir / "xgb_model.pkl"),

        "ANN":
            joblib.load(model_dir / "ann_model.pkl")

    }

    tfidf = joblib.load(model_dir / "tfidf.pkl")

    encoder = joblib.load(model_dir / "label_encoder.pkl")

    vader = SentimentIntensityAnalyzer()

    return models, tfidf, encoder, vader


models, tfidf, encoder, vader = load_models()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("Model Selection")

selected_model = st.sidebar.selectbox(

    "Choose Model",

    [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "SVM",
        "VADER",
        "XGBoost",
        "ANN"
    ]

)

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

        results = []

        # ------------------------------------
        # Run every ML model
        # ------------------------------------

        for model_name, model in models.items():

            prediction = None
            confidence = None

            if model_name == "XGBoost":

                pred = model.predict(vector)
                prediction = encoder.inverse_transform(pred)[0]

            else:

                prediction = model.predict(vector)[0]

            probabilities = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(vector)[0]
                confidence = float(probabilities.max())

            elif hasattr(model, "decision_function"):

                import numpy as np

                score = model.decision_function(vector)

                if score.ndim == 1:
                    confidence = float(
                        1 / (1 + np.exp(-abs(score[0])))
                    )

                else:
                    exp_scores = np.exp(score)
                    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                    confidence = float(probs.max())

            else:
                confidence = 0.0

            results.append({

                "Model": model_name,
                "Prediction": prediction,
                "Confidence": confidence

            })

        # ------------------------------------
        # Run VADER
        # ------------------------------------

        vader_score = vader.polarity_scores(review)

        if vader_score["compound"] >= 0.05:
            vader_prediction = "positive"

        elif vader_score["compound"] <= -0.05:
            vader_prediction = "negative"

        else:
            vader_prediction = "neutral"

        results.append({

            "Model": "VADER",
            "Prediction": vader_prediction,
            "Confidence": abs(vader_score["compound"])

        })

        # ------------------------------------
        # Convert to dataframe
        # ------------------------------------

        result_df = pd.DataFrame(results)

        result_df["Confidence %"] = (
            result_df["Confidence"] * 100
        ).round(2)

        result_df = result_df.sort_values(
            by="Confidence",
            ascending=False
        ).reset_index(drop=True)

        # ------------------------------------
        # Best model
        # ------------------------------------

        best = result_df.iloc[0]

        st.markdown("## 🏆 Best Performing Model")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Model",
            best["Model"]
        )

        col2.metric(
            "Prediction",
            best["Prediction"].upper()
        )

        col3.metric(
            "Confidence",
            f'{best["Confidence"]:.2%}'
        )

        st.success(
            f"**{best['Model']}** produced the highest confidence "
            f"({best['Confidence']:.2%}) for this review."
        )

        st.markdown("---")

        # ------------------------------------
        # Comparison Table
        # ------------------------------------

        st.subheader("Model Comparison")

        st.dataframe(

            result_df[
                ["Model", "Prediction", "Confidence %"]
            ],

            use_container_width=True

        )

        # ------------------------------------
        # Confidence Chart
        # ------------------------------------

        st.subheader("Confidence Comparison")

        chart_df = result_df.copy()

        chart_df = chart_df.set_index("Model")

        st.bar_chart(
            chart_df["Confidence %"],
            use_container_width=True
        )

        # ------------------------------------
        # Agreement Summary
        # ------------------------------------

        st.subheader("Prediction Agreement")

        agree = (
            result_df.groupby("Prediction")
            .size()
            .reset_index(name="Votes")
            .sort_values(
                by="Votes",
                ascending=False
            )
        )

        st.dataframe(
            agree,
            use_container_width=True
        )

        winner = agree.iloc[0]

        st.info(

            f"Majority Vote Prediction: **{winner['Prediction'].upper()}** "
            f"({winner['Votes']} out of {len(result_df)} models)"

        )

        # ------------------------------------
        # Probability of Selected Model
        # ------------------------------------

        if selected_model != "VADER":

            model = models[selected_model]

            if hasattr(model, "predict_proba"):

                probs = model.predict_proba(vector)[0]

                prob_df = pd.DataFrame({

                    "Sentiment": encoder.classes_,
                    "Probability": probs

                })

                st.subheader(
                    f"{selected_model} Probability Distribution"
                )

                st.bar_chart(

                    prob_df,

                    x="Sentiment",

                    y="Probability",

                    use_container_width=True

                )

# -------------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.write("### Available Models")

st.sidebar.write("""
- Logistic Regression
- Naive Bayes
- Random Forest
- SVM
- VADER
- XGBoost
- ANN
""")

st.sidebar.markdown("---")

st.sidebar.info(
"""
This application loads pre-trained models.
No model training occurs during startup,
making the application much faster.
"""
)