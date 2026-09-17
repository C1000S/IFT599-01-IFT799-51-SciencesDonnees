import os

import pandas as pd

from distances import pairwise_overlap

OUT_DIR = "figures/method1"

X = pd.read_csv("data_cache/X_reduced.csv", index_col=0)
y = pd.read_csv("data_cache/y.csv", index_col=0)["Class"]

os.makedirs(OUT_DIR, exist_ok=True)

for metric in ["euclidean", "mahalanobis"]:
    print(f"\n=== Metrique : {metric} ===")
    overlap_matrix, inter_matrix, intra = pairwise_overlap(X, y, metric=metric)

    print("\nDistance intra-classe :")
    print(intra.round(3))

    print("\nMatrice Overlap(C1, C2) :")
    print(overlap_matrix.round(3))

    print("\nPaires bien separees (Overlap < 1) :")
    classes = overlap_matrix.columns
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1:]:
            v = overlap_matrix.loc[c1, c2]
            status = "separees" if v < 1 else "chevauchement"
            print(f"  {c1} vs {c2} : Overlap = {v:.3f} -> {status}")

    intra.round(4).to_csv(f"{OUT_DIR}/intra_{metric}.csv", header=["dist_intra"])
    overlap_matrix.round(4).to_csv(f"{OUT_DIR}/overlap_{metric}.csv")
    inter_matrix.round(4).to_csv(f"{OUT_DIR}/inter_{metric}.csv")
