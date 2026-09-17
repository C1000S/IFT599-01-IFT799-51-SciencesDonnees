import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def class_center(X):
    return X.mean(axis=0, keepdims=True)


def dist_intra(X, metric="euclidean", VI=None):
    center = class_center(X)
    kwargs = {"VI": VI} if metric == "mahalanobis" else {}
    return cdist(X, center, metric=metric, **kwargs).max()


def dist_inter(X1, X2, metric="euclidean", VI=None):
    c1, c2 = class_center(X1), class_center(X2)
    kwargs = {"VI": VI} if metric == "mahalanobis" else {}
    d_1_to_c2 = cdist(X1, c2, metric=metric, **kwargs).min()
    d_2_to_c1 = cdist(X2, c1, metric=metric, **kwargs).min()
    return min(d_1_to_c2, d_2_to_c1)


def overlap(intra1, intra2, inter):
    if inter == 0:
        return np.inf
    return (intra1 + intra2) / (2 * inter)


def pairwise_overlap(X, y, metric="euclidean"):
    classes = sorted(y.unique())
    VI = None
    if metric == "mahalanobis":
        cov = np.cov(X.values, rowvar=False)
        VI = np.linalg.inv(cov)

    class_data = {c: X[y == c].values for c in classes}
    intra = {c: dist_intra(class_data[c], metric=metric, VI=VI) for c in classes}

    overlap_matrix = pd.DataFrame(index=classes, columns=classes, dtype=float)
    inter_matrix = pd.DataFrame(index=classes, columns=classes, dtype=float)
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1:]:
            inter = dist_inter(class_data[c1], class_data[c2], metric=metric, VI=VI)
            ov = overlap(intra[c1], intra[c2], inter)
            overlap_matrix.loc[c1, c2] = ov
            overlap_matrix.loc[c2, c1] = ov
            inter_matrix.loc[c1, c2] = inter
            inter_matrix.loc[c2, c1] = inter

    return overlap_matrix, inter_matrix, pd.Series(intra, name="dist_intra")
