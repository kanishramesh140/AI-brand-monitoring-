import pandas as pd
import numpy as np

def predict_reputation(data):

    sentiment_map={
        "Positive":1,
        "Negative":-1
    }

    data["score"]=data["sentiment"].map(sentiment_map)

    trend=data.groupby("brand")["score"].sum().reset_index()

    trend["future_score"]=trend["score"]+np.random.randint(-2,3,len(trend))

    return trend