import os

import pandas as pd

from plotting import scatter_with_marginals
from variable_selection import select_all_local_pairs, select_global_pair

FIG_DIR = "figures/method2_original_vars"

X = pd.read_csv("data_cache/X_reduced.csv", index_col=0)
y = pd.read_csv("data_cache/y.csv", index_col=0)["Class"]

os.makedirs(FIG_DIR, exist_ok=True)
rows = []

print("=== Selection globale (5 classes simultanement) ===")
global_pair, global_score = select_global_pair(X, y)
print(f"Meilleure paire : {global_pair}, score (Overlap moyen) = {global_score:.4f}")
scatter_with_marginals(
    X, y, *global_pair,
    title=f"Toutes les classes - Overlap moyen={global_score:.2f}",
    save_path=f"{FIG_DIR}/global.png",
)
rows.append({"comparaison": "toutes classes", "var1": global_pair[0], "var2": global_pair[1], "overlap": global_score})

print("\n=== Selection locale (par paire de classes) ===")
local_results = select_all_local_pairs(X, y)
for (c1, c2), (pair, score) in local_results.items():
    print(f"{c1} vs {c2} : paire = {pair}, Overlap = {score:.4f}")
    scatter_with_marginals(
        X, y, *pair,
        title=f"{c1} vs {c2} - Overlap={score:.2f}",
        save_path=f"{FIG_DIR}/{c1}_vs_{c2}.png",
        classes=[c1, c2],
    )
    rows.append({"comparaison": f"{c1} vs {c2}", "var1": pair[0], "var2": pair[1], "overlap": score})

summary = pd.DataFrame(rows)
summary.to_csv(f"{FIG_DIR}/summary.csv", index=False)
print(f"\nFigures et tableau sauvegardes dans {FIG_DIR}/")
