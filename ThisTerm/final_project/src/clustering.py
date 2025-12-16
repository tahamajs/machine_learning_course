from typing import Tuple, Dict, Any
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> Dict[str, Any]:
    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels) if k > 1 and len(X) > k else float("nan")
    return {"model": km, "labels": labels, "silhouette": score}


def run_agglomerative(X: np.ndarray, k: int) -> Dict[str, Any]:
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X)
    score = silhouette_score(X, labels) if k > 1 and len(X) > k else float("nan")
    return {"model": agg, "labels": labels, "silhouette": score}




