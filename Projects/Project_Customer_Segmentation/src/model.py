from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

def fit_kmeans(n_clusters, X, preprocessor):
    kmeans_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=9, verbose=0))
    ])
    kmeans_pipeline.fit(X)
    return kmeans_pipeline.named_steps["cluster"].inertia_
