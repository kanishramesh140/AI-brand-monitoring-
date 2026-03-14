import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("../data/reviews_dataset.csv")

X = data["text"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

category_model = LogisticRegression()
category_model.fit(X_vec, data["category"])

sentiment_model = LogisticRegression()
sentiment_model.fit(X_vec, data["sentiment"])

fake_model = LogisticRegression()
fake_model.fit(X_vec, data["fake"])

joblib.dump(
{
"vectorizer":vectorizer,
"category_model":category_model,
"sentiment_model":sentiment_model,
"fake_model":fake_model
},
"../models/models.pkl"
)

print("Model training completed")