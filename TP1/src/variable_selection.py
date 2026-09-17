import itertools

import numpy as np

from distances import pairwise_overlap


def global_score(X, y, cols):
    """Critere 'toutes les classes' : Overlap moyen sur les 10 paires de classes.
    Plus petit = meilleure separation globale."""
    overlap_matrix, _, _ = pairwise_overlap(X[cols], y, metric="euclidean")
    vals = overlap_matrix.values[np.triu_indices_from(overlap_matrix.values, k=1)]
    return vals.mean()


def local_score(X, y, cols, class1, class2):
    """Critere 'une paire de classes' : Overlap(class1, class2) uniquement."""
    mask = y.isin([class1, class2])
    overlap_matrix, _, _ = pairwise_overlap(X.loc[mask, cols], y[mask], metric="euclidean")
    return overlap_matrix.loc[class1, class2]


def sfs_select_pair(X, y, eval_fn, candidates=None):
    """Sequential Forward Selection, arret a 2 variables (on cherche une paire).
    eval_fn(X, y, cols) -> score scalaire, plus petit = meilleur."""
    candidates = list(candidates if candidates is not None else X.columns)

    scores_1 = {v: eval_fn(X, y, [v]) for v in candidates}
    v1 = min(scores_1, key=scores_1.get)

    remaining = [v for v in candidates if v != v1]
    scores_2 = {v: eval_fn(X, y, [v1, v]) for v in remaining}
    v2 = min(scores_2, key=scores_2.get)

    return (v1, v2), scores_2[v2]


def select_global_pair(X, y, candidates=None):
    return sfs_select_pair(X, y, global_score, candidates=candidates)


def select_all_local_pairs(X, y, candidates=None):
    results = {}
    for c1, c2 in itertools.combinations(sorted(y.unique()), 2):
        eval_fn = lambda X_, y_, cols, c1=c1, c2=c2: local_score(X_, y_, cols, c1, c2)
        pair, score = sfs_select_pair(X, y, eval_fn, candidates=candidates)
        results[(c1, c2)] = (pair, score)
    return results
