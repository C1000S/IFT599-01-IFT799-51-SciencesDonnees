import numpy as np
import pandas as pd

DATA_PATH = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
LABELS_PATH = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"


def load_data(data_path=DATA_PATH, labels_path=LABELS_PATH):
    X = pd.read_csv(data_path, index_col=0)
    y = pd.read_csv(labels_path, index_col=0)["Class"]
    return X, y


def filter_near_zero_variance(X, threshold=1e-8):
    variances = X.var(axis=0)
    keep = variances[variances > threshold].index
    return X[keep]


def select_random_subset(X, n_vars, seed=42):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(X.columns, size=n_vars, replace=False)
    return X[list(chosen)]


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)
