# ==========================================================
# AI BRAND INTELLIGENCE COMMAND CENTER (Streamlit Cloud Ready)
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# FIXED __file__
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.language_reply import detect_language, auto_reply
from scripts.reputation import reputation_score
from scripts.prediction import predict_reputation
from scripts.galaxy_visualization import generate_galaxy
from dashboard.theme import apply_theme


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Brand Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# APPLY THEME
# ==========================================================

apply_theme()


# ==========================================================
# MOBILE RESPONSIVE
# ==========================================================

st.markdown("""
<style>
@media (max-width:768px){
.stPlotlyChart{
height:350px !important;
}
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# PATHS
# ==========================================================

DATA_DIR = "data"
MODEL_DIR = "models"

DATA_PATH = os.path.join(DATA_DIR, "review_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "models.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# AUTO CREATE DATA
# ==========================================================

if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):

    st.warning("Dataset or models missing. Creating them...")

    import scripts.train_model


# ==========================================================
# LOAD DATA
# ==========================================================

data = pd.read_csv(DATA_PATH)

models = joblib.load(MODEL_PATH)

vectorizer = models["vectorizer"]
category_model = models["category_model"]
sentiment_model = models["sentiment_model"]
fake_model = models["fake_model"]


# ==========================================================
# HEADER
# ==========================================================

st.title("🚀 AI Brand Intelligence Command Center")

st.write("AI powered brand monitoring platform")


# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

st.sidebar.header("Filters")

platform_filter = st.sidebar.multiselect(
    "Platform",
    data["platform"].unique(),
    default=data["platform"].unique()
)

brand_filter = st.sidebar.multiselect(
    "Brand",
    data["brand"].unique(),
    default=data["brand"].unique()
)

filtered = data[
    (data["platform"].isin(platform_filter)) &
    (data["brand"].isin(brand_filter))
]


# ==========================================================
# SENTIMENT DISTRIBUTION
# ==========================================================

st.subheader("Customer Sentiment Distribution")

fig = px.pie(
    filtered,
    names="sentiment",
    title="Sentiment Distribution"
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# BRAND WAR ANALYTICS
# ==========================================================

st.subheader("Brand War Analytics")

brand_sentiment = filtered.groupby(
    ["brand", "sentiment"]
).size().reset_index(name="count")

fig2 = px.bar(
    brand_sentiment,
    x="brand",
    y="count",
    color="sentiment",
    barmode="group",
    title="Brand Competition Sentiment"
)

st.plotly_chart(fig2, use_container_width=True)


# ==========================================================
# PLATFORM COMPLAINT MAP
# ==========================================================

st.subheader("Platform Complaint Map")

complaints = filtered[filtered["sentiment"] == "Negative"]

fig3 = px.histogram(
    complaints,
    x="platform",
    color="brand",
    title="Complaints by Platform"
)

st.plotly_chart(fig3, use_container_width=True)


# ==========================================================
# REPUTATION SCORE
# ==========================================================

st.subheader("Brand Reputation Score")

brands = filtered["brand"].unique()

selected_brand = st.selectbox("Select Brand", brands)

brand_data = filtered[filtered["brand"] == selected_brand]

texts = brand_data["text"]

vectors = vectorizer.transform(texts)

sentiments = list(sentiment_model.predict(vectors))

score = reputation_score(sentiments)

st.metric("Reputation Score", f"{score}%")


# ==========================================================
# FUTURE REPUTATION PREDICTION
# ==========================================================

st.subheader("Reputation Prediction")

forecast = predict_reputation(filtered)

fig4 = px.bar(
    forecast,
    x="brand",
    y="future_score",
    title="Future Reputation Forecast"
)

st.plotly_chart(fig4, use_container_width=True)


# ==========================================================
# 🌌 BRAND REPUTATION GALAXY (3D UPGRADE)
# ==========================================================

st.subheader("🌌 AI Brand Reputation Galaxy")

st.write(
"Each brand appears as a planet. "
"Size represents reputation strength. "
"Color indicates sentiment."
)

galaxy_fig = generate_galaxy(filtered)

st.plotly_chart(galaxy_fig, use_container_width=True)


# ==========================================================
# 🚨 AI ANOMALY DETECTION (CRISIS MONITOR)
# ==========================================================

st.subheader("🚨 Brand Crisis Detector")

negative_counts = filtered.groupby("brand")["sentiment"].apply(
    lambda x: (x == "Negative").sum()
)

threshold = negative_counts.mean() + negative_counts.std()

alerts = negative_counts[negative_counts > threshold]

if len(alerts) > 0:

    st.error("⚠️ Potential Brand Crisis Detected!")

    for brand, count in alerts.items():

        st.write(f"🚨 {brand} showing unusual negative spike ({count})")

else:

    st.success("No brand crisis detected")


# ==========================================================
# COMPLAINT TABLE
# ==========================================================

st.subheader("Recent Complaints")

st.dataframe(
    complaints[["platform", "brand", "text", "category"]],
    use_container_width=True
)


# ==========================================================
# AI REVIEW ANALYZER
# ==========================================================

st.subheader("Analyze New Review")

text = st.text_area("Enter customer review")

if st.button("Analyze"):

    lang = detect_language(text)

    vec = vectorizer.transform([text])

    category = category_model.predict(vec)[0]

    sentiment = sentiment_model.predict(vec)[0]

   text_lower = text.lower()

if (
    "!!!" in text
    or "100%" in text
    or "must buy" in text_lower
    or "everyone must buy" in text_lower
    or "best product ever" in text_lower
    or "best phone ever" in text_lower
    or "100% recommended" in text_lower
    or "guaranteed results" in text_lower
    or "life changing" in text_lower
    or "amazing product" in text_lower
    or "perfect product" in text_lower
    or "top quality" in text_lower
    or "highest quality" in text_lower
    or "super product" in text_lower
    or "awesome product" in text_lower
    or "incredible product" in text_lower
    or "fantastic product" in text_lower
    or "buy this now" in text_lower
    or "limited offer" in text_lower
    or "don't miss this" in text_lower
    or "highly recommend" in text_lower
    or "highly recommended" in text_lower
    or "worth every penny" in text_lower
    or "totally worth it" in text_lower
    or "best in the market" in text_lower
    or "number one product" in text_lower
    or "unbelievable" in text_lower
    or "outstanding" in text_lower
    or "no.1 brand" in text_lower
    or "best ever" in text_lower
    or "love this product" in text_lower
    or "absolutely perfect" in text_lower
    or "perfect in every way" in text_lower
):
    fake = 1
elif fake_model is not None:
    fake = fake_model.predict(vec)[0]
else:
    fake = 0

    reply = auto_reply(lang, sentiment)

    st.write("Language:", lang)
    st.write("Category:", category)
    st.write("Sentiment:", sentiment)
    st.write("Fake Review:", "Yes" if fake == 1 else "No")
    st.write("Auto Reply:", reply)