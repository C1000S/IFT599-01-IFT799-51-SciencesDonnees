import os

from data_loading import filter_near_zero_variance, load_data, select_random_subset, standardize

N_VARS = 200
SEED = 42
OUT_DIR = "data_cache"


def main():
    X, y = load_data()
    print(f"Donnees brutes : {X.shape[0]} patients x {X.shape[1]} genes")

    X = filter_near_zero_variance(X)
    print(f"Apres filtrage variance quasi nulle : {X.shape[1]} genes restants")

    X = select_random_subset(X, N_VARS, seed=SEED)
    X = standardize(X)
    print(f"Sous-ensemble final : {X.shape[1]} genes (seed={SEED})")

    os.makedirs(OUT_DIR, exist_ok=True)
    X.to_csv(f"{OUT_DIR}/X_reduced.csv")
    y.to_csv(f"{OUT_DIR}/y.csv")
    with open(f"{OUT_DIR}/selected_genes.txt", "w") as f:
        f.write("\n".join(X.columns))

    print(f"Sauvegarde dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
