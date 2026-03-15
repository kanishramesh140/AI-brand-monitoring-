# ==========================================================
# BRAND REPUTATION SCORE CALCULATION
# ==========================================================

def reputation_score(sentiments):

    # convert to list if numpy array
    sentiments = list(sentiments)

    total = len(sentiments)

    if total == 0:
        return 0

    positive = sentiments.count("Positive")
    negative = sentiments.count("Negative")

    score = ((positive - negative) / total) * 100

    return round(score, 2)