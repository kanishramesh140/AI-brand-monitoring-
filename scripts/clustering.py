from sklearn.cluster import KMeans

def cluster_complaints(vectors):

    model = KMeans(n_clusters=3)

    clusters = model.fit_predict(vectors)

    return clusters