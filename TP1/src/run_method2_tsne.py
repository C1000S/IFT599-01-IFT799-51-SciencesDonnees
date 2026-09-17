import os

import pandas as pd
from sklearn.manifold import TSNE

from distances import pairwise_overlap
from plotting import scatter_with_marginals

FIG_DIR = "figures/method2_tsne"
SEED = 42

X = pd.read_csv("data_cache/X_reduced.csv", index_col=0)
y = pd.read_csv("data_cache/y.csv", index_col=0)["Class"]


def run_tsne(X_subset):
    tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(X_subset.values)
    return pd.DataFrame(emb, index=X_subset.index, columns=["tSNE1", "tSNE2"])


os.makedirs(FIG_DIR, exist_ok=True)
rows = []

print("=== t-SNE global (5 classes simultanement) ===")
X_emb_global = run_tsne(X)
overlap_matrix, _, _ = pairwise_overlap(X_emb_global, y, metric="euclidean")
mean_overlap = overlap_matrix.values[overlap_matrix.notna().values].mean()
print(f"Overlap moyen sur l'embedding global = {mean_overlap:.4f}")
scatter_with_marginals(
    X_emb_global, y, "tSNE1", "tSNE2",
    title=f"t-SNE - Toutes les classes - Overlap moyen={mean_overlap:.2f}",
    save_path=f"{FIG_DIR}/global.png",
)
rows.append({"comparaison": "toutes classes", "var1": "tSNE1", "var2": "tSNE2", "overlap": mean_overlap})

print("\n=== t-SNE local (embedding recalcule pour chaque paire de classes) ===")
classes = sorted(y.unique())
for i, c1 in enumerate(classes):
    for c2 in classes[i + 1:]:
        mask = y.isin([c1, c2])
        X_emb_local = run_tsne(X[mask])
        overlap_matrix, _, _ = pairwise_overlap(X_emb_local, y[mask], metric="euclidean")
        score = overlap_matrix.loc[c1, c2]
        print(f"{c1} vs {c2} : Overlap = {score:.4f}")
        scatter_with_marginals(
            X_emb_local, y[mask], "tSNE1", "tSNE2",
            title=f"t-SNE - {c1} vs {c2} - Overlap={score:.2f}",
            save_path=f"{FIG_DIR}/{c1}_vs_{c2}.png",
            classes=[c1, c2],
        )
        rows.append({"comparaison": f"{c1} vs {c2}", "var1": "tSNE1", "var2": "tSNE2", "overlap": score})

summary = pd.DataFrame(rows)
summary.to_csv(f"{FIG_DIR}/summary.csv", index=False)
print(f"\nFigures et tableau sauvegardes dans {FIG_DIR}/")
