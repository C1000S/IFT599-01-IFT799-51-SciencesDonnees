import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CLASS_ORDER = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]
CLASS_STYLE = {
    "BRCA": dict(color="#2a78d6", marker="o"),
    "COAD": dict(color="#eb6834", marker="s"),
    "KIRC": dict(color="#1baf7a", marker="^"),
    "LUAD": dict(color="#eda100", marker="D"),
    "PRAD": dict(color="#e87ba4", marker="v"),
}


def scatter_two_vars(X, y, var1, var2, title, save_path, classes=None):
    classes = classes if classes is not None else CLASS_ORDER
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in classes:
        mask = y == cls
        style = CLASS_STYLE[cls]
        ax.scatter(
            X.loc[mask, var1], X.loc[mask, var2],
            label=cls, s=28, alpha=0.75,
            edgecolors="white", linewidths=0.4,
            **style,
        )
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def scatter_with_marginals(X, y, var1, var2, title, save_path, classes=None):
    """Nuage de points + histogrammes 1D en marge (equivalent d'un seaborn.jointplot),
    cf. Methode 2 point 2(b) de l'enonce : distribution de chaque variable, par classe."""
    classes = classes if classes is not None else CLASS_ORDER

    fig = plt.figure(figsize=(6.5, 6))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    for cls in classes:
        mask = y == cls
        style = CLASS_STYLE[cls]
        ax_scatter.scatter(
            X.loc[mask, var1], X.loc[mask, var2],
            label=cls, s=28, alpha=0.75,
            edgecolors="white", linewidths=0.4,
            **style,
        )
        ax_histx.hist(X.loc[mask, var1], bins=20, color=style["color"], alpha=0.5)
        ax_histy.hist(
            X.loc[mask, var2], bins=20, color=style["color"], alpha=0.5,
            orientation="horizontal",
        )

    ax_scatter.set_xlabel(var1)
    ax_scatter.set_ylabel(var2)
    ax_scatter.legend(frameon=False, fontsize=9)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    for ax in (ax_histx, ax_histy):
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
