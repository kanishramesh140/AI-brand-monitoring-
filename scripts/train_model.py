
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ======================
# Relative paths
# ======================
DATA_DIR = "data"
MODEL_DIR = "models"
CSV_FILE = "review_dataset.csv"
DATA_PATH = os.path.join(DATA_DIR, CSV_FILE)
MODEL_PATH = os.path.join(MODEL_DIR, "models.pkl")

# Create folders if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# Dataset (ensure at least 2 classes in 'fake')
# ======================
data = [
    ["Twitter","Samsung","Samsung battery draining fast","Product","Negative",0],
    ["Twitter","Apple","I love the new iPhone camera","Feedback","Positive",0],
    ["Twitter","Jio","Jio internet very slow today","Service","Negative",0],
    ["Twitter","Airtel","Airtel network problem in my area","Service","Negative",1],

    ["Amazon","Samsung","This phone is amazing great display","Feedback","Positive",0],
    ["Amazon","OnePlus","Battery problem after update","Product","Negative",0],
    ["Amazon","Apple","Excellent product worth buying","Feedback","Positive",1],
    ["Amazon","Realme","Phone overheating while charging","Product","Negative",0],

    ["Flipkart","Realme","Phone heating issue while gaming","Product","Negative",0],
    ["Flipkart","Vivo","Camera quality is very good","Feedback","Positive",0],
    ["Flipkart","Samsung","Delivery was fast and packaging good","Delivery","Positive",0],
    ["Flipkart","Apple","Product price too high for features","Product","Negative",1],

    ["PlayStore","Jio","Jio app crashes frequently","Service","Negative",0],
    ["PlayStore","Google","Great update very smooth","Feedback","Positive",0],
    ["PlayStore","Airtel","App login issue after update","Service","Negative",0],
    ["PlayStore","Samsung","Samsung health app works perfectly","Feedback","Positive",0],

    ["YouTube","Samsung","This phone review is amazing","Feedback","Positive",0],
    ["YouTube","Apple","Overpriced phone not worth it","Product","Negative",0],
    ["YouTube","Realme","Great performance for gaming","Feedback","Positive",0],
    ["YouTube","Vivo","Camera stabilization is impressive","Feedback","Positive",1]
]

# Convert to DataFrame and save CSV
df = pd.DataFrame(data, columns=["platform","brand","text","category","sentiment","fake"])
df.to_csv(DATA_PATH, index=False)
print(f"Dataset created at {DATA_PATH}")

# ======================
# Vectorization
# ======================
X = df["text"]
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# ======================
# Train Models
# ======================
category_model = LogisticRegression(max_iter=500)
category_model.fit(X_vec, df["category"])

sentiment_model = LogisticRegression(max_iter=500)
sentiment_model.fit(X_vec, df["sentiment"])

# Only train fake_model if 2 classes exist
if len(df["fake"].unique()) > 1:
    fake_model = LogisticRegression(max_iter=500)
    fake_model.fit(X_vec, df["fake"])
else:
    fake_model = None
    print("Skipping fake model: only 1 class found")

# ======================
# Save models
# ======================
joblib.dump({
    "vectorizer": vectorizer,
    "category_model": category_model,
    "sentiment_model": sentiment_model,
    "fake_model": fake_model
}, MODEL_PATH)

print(f"Models saved at {MODEL_PATH}")