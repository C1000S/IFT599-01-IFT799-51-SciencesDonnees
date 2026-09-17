import os

import pandas as pd
from sklearn.decomposition import PCA

from plotting import scatter_with_marginals
from variable_selection import select_all_local_pairs, select_global_pair

N_COMPONENTS = 15
FIG_DIR = "figures/method2_pca"

X = pd.read_csv("data_cache/X_reduced.csv", index_col=0)
y = pd.read_csv("data_cache/y.csv", index_col=0)["Class"]

pca = PCA(n_components=N_COMPONENTS, random_state=42)
pcs = pca.fit_transform(X)
pc_cols = [f"PC{i+1}" for i in range(N_COMPONENTS)]
X_pca = pd.DataFrame(pcs, index=X.index, columns=pc_cols)

print(f"Variance expliquee par composante : {pca.explained_variance_ratio_.round(3)}")
print(f"Variance expliquee cumulee (15 PC) : {pca.explained_variance_ratio_.sum():.3f}")

os.makedirs(FIG_DIR, exist_ok=True)
rows = []

print("\n=== Selection globale (5 classes simultanement), sur composantes ACP ===")
global_pair, global_score = select_global_pair(X_pca, y)
print(f"Meilleure paire : {global_pair}, score (Overlap moyen) = {global_score:.4f}")
scatter_with_marginals(
    X_pca, y, *global_pair,
    title=f"ACP - Toutes les classes - Overlap moyen={global_score:.2f}",
    save_path=f"{FIG_DIR}/global.png",
)
rows.append({"comparaison": "toutes classes", "var1": global_pair[0], "var2": global_pair[1], "overlap": global_score})

print("\n=== Selection locale (par paire de classes), sur composantes ACP ===")
local_results = select_all_local_pairs(X_pca, y)
for (c1, c2), (pair, score) in local_results.items():
    print(f"{c1} vs {c2} : paire = {pair}, Overlap = {score:.4f}")
    scatter_with_marginals(
        X_pca, y, *pair,
        title=f"ACP - {c1} vs {c2} - Overlap={score:.2f}",
        save_path=f"{FIG_DIR}/{c1}_vs_{c2}.png",
        classes=[c1, c2],
    )
    rows.append({"comparaison": f"{c1} vs {c2}", "var1": pair[0], "var2": pair[1], "overlap": score})

summary = pd.DataFrame(rows)
summary.to_csv(f"{FIG_DIR}/summary.csv", index=False)
print(f"\nFigures et tableau sauvegardes dans {FIG_DIR}/")
