import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# Define paths
# ==========================================
PROJECT_DIR = r"C:\Users\Kanish\OneDrive\Desktop\brand"  # your project folder
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
CSV_FILE = "review_dataset.csv"
DATA_PATH = os.path.join(DATA_DIR, CSV_FILE)

# Create folders if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# Dataset with at least 2 classes in 'fake'
# ==========================================
data = [
    ["Twitter","Samsung","Samsung battery draining fast","Product","Negative",0],
    ["Twitter","Apple","I love the new iPhone camera","Feedback","Positive",0],
    ["Twitter","Jio","Jio internet very slow today","Service","Negative",0],
    ["Twitter","Airtel","Airtel network problem in my area","Service","Negative",1],  # <-- 1 class

    ["Amazon","Samsung","This phone is amazing great display","Feedback","Positive",0],
    ["Amazon","OnePlus","Battery problem after update","Product","Negative",0],
    ["Amazon","Apple","Excellent product worth buying","Feedback","Positive",1],  # <-- 1 class
    ["Amazon","Realme","Phone overheating while charging","Product","Negative",0],

    ["Flipkart","Realme","Phone heating issue while gaming","Product","Negative",0],
    ["Flipkart","Vivo","Camera quality is very good","Feedback","Positive",0],
    ["Flipkart","Samsung","Delivery was fast and packaging good","Delivery","Positive",0],
    ["Flipkart","Apple","Product price too high for features","Product","Negative",1],  # <-- 1 class

    ["PlayStore","Jio","Jio app crashes frequently","Service","Negative",0],
    ["PlayStore","Google","Great update very smooth","Feedback","Positive",0],
    ["PlayStore","Airtel","App login issue after update","Service","Negative",0],
    ["PlayStore","Samsung","Samsung health app works perfectly","Feedback","Positive",0],

    ["YouTube","Samsung","This phone review is amazing","Feedback","Positive",0],
    ["YouTube","Apple","Overpriced phone not worth it","Product","Negative",0],
    ["YouTube","Realme","Great performance for gaming","Feedback","Positive",0],
    ["YouTube","Vivo","Camera stabilization is impressive","Feedback","Positive",1],  # <-- 1 class
]

# Convert to DataFrame
df = pd.DataFrame(
    data,
    columns=["platform", "brand", "text", "category", "sentiment", "fake"]
)

# Save CSV (overwrite if exists)
df.to_csv(DATA_PATH, index=False, encoding='utf-8')
print(f"Dataset created successfully at: {DATA_PATH}")
print(df.head())

# ==========================================
# Features and Vectorization
# ==========================================
X = df["text"]
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# ==========================================
# Train Models
# ==========================================
category_model = LogisticRegression(max_iter=500)
category_model.fit(X_vec, df["category"])

sentiment_model = LogisticRegression(max_iter=500)
sentiment_model.fit(X_vec, df["sentiment"])

# Fake model: only train if there are 2 classes
if len(df["fake"].unique()) > 1:
    fake_model = LogisticRegression(max_iter=500)
    fake_model.fit(X_vec, df["fake"])
else:
    print("Skipping fake model: only one class found")
    fake_model = None

# ==========================================
# Save models
# ==========================================
model_file_path = os.path.join(MODEL_DIR, "models.pkl")
joblib.dump(
    {
        "vectorizer": vectorizer,
        "category_model": category_model,
        "sentiment_model": sentiment_model,
        "fake_model": fake_model
    },
    model_file_path
)

print(f"Models trained and saved successfully at: {model_file_path}")