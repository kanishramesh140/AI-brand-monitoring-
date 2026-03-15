# ==========================================================
# AI BRAND INTELLIGENCE COMMAND CENTER (Streamlit Cloud Ready)
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys

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
# APPLY DASHBOARD THEME
# ==========================================================

apply_theme()


# ==========================================================
# MOBILE RESPONSIVE CSS
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
# STREAMLIT CLOUD FRIENDLY PATHS
# ==========================================================

DATA_DIR = "data"
MODEL_DIR = "models"

DATA_PATH = os.path.join(DATA_DIR, "review_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "models.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# AUTO CREATE DATASET & MODELS IF MISSING
# ==========================================================

if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
    st.warning("Dataset or models missing. Creating them...")
    import scripts.train_model


# ==========================================================
# LOAD DATA AND MODELS
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

selected_brand = st.selectbox(
    "Select Brand",
    brands
)

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
# 🌌 BRAND REPUTATION GALAXY
# ==========================================================

st.subheader("🌌 AI Brand Reputation Galaxy")

st.write(
    "Each brand appears as a planet. "
    "Size represents reputation strength. "
    "Color indicates positive or negative sentiment."
)

galaxy_fig = generate_galaxy(filtered)

st.dataframe(filtered)

st.plotly_chart(galaxy_fig, use_container_width=True)


# ==========================================================
# ⭐ SMALL UPGRADE: 3D GALAXY VISUALIZATION
# ==========================================================

st.subheader("🪐 3D Brand Galaxy View")

brands = filtered["brand"].unique()

x = np.random.rand(len(brands))
y = np.random.rand(len(brands))
z = np.random.rand(len(brands))

sizes = []

for b in brands:
    brand_data = filtered[filtered["brand"] == b]
    vec = vectorizer.transform(brand_data["text"])
    sentiments = sentiment_model.predict(vec)
    score = reputation_score(sentiments)
    sizes.append(score)

fig3d = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers+text',
    text=brands,
    marker=dict(
        size=np.array(sizes)/2,
        color=sizes,
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig3d.update_layout(title="3D Brand Reputation Galaxy")

st.plotly_chart(fig3d, use_container_width=True)


# ==========================================================
# POWERFUL IMPROVEMENT: REAL-TIME BRAND MONITORING
# ==========================================================

st.subheader("📡 Live Brand Sentiment Monitor")

placeholder = st.empty()

live_data = filtered.sample(min(20, len(filtered)))

live_counts = live_data["sentiment"].value_counts()

live_fig = px.bar(
    x=live_counts.index,
    y=live_counts.values,
    labels={"x": "Sentiment", "y": "Mentions"},
    title="Live Sentiment Snapshot"
)

placeholder.plotly_chart(live_fig, use_container_width=True)


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

    if fake_model is not None:
        fake = fake_model.predict(vec)[0]
    else:
        fake = 0

    reply = auto_reply(lang, sentiment)

    st.write("Language:", lang)
    st.write("Category:", category)
    st.write("Sentiment:", sentiment)
    st.write("Fake Review:", "Yes" if fake == 1 else "No")
    st.write("Auto Reply:", reply)