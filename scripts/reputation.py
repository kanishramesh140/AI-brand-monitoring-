def reputation_score(sentiments):

    positive = sentiments.count("Positive")
    negative = sentiments.count("Negative")

    total = positive + negative

    if total==0:
        return 0

    score = (positive/total)*100

    return round(score,2)