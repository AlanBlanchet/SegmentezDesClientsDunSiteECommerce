import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from tqdm import tqdm


def view_projection(
    projectors,
    X,
    hue=None,
    p=1.0,
    watch_params=["gamma", "alpha", "kernel", "perplexity"],
    models=None,
):
    n = int(X.shape[0] * p)
    X = X[:n]

    if hue is not None:
        hue = hue[:n]

    max_cols = 3

    model_projectors = []
    for projector in projectors:
        if models is not None:
            for model in models:
                model_projectors.append((projector, model))
        else:
            model_projectors.append((projector, None))

    lines = math.ceil(len(model_projectors) / max_cols)

    fig, axs = plt.subplots(
        lines, min(len(model_projectors), max_cols), figsize=(24, lines * 10)
    )
    fig.suptitle(
        f"All model evaluations from {p*100}% of the dataset ({X.shape[0]} datapoints)"
    )

    for (projector, model), ax in tqdm(zip(model_projectors, np.array(axs).flatten())):
        if model is not None:
            name = type(model).__name__ + " "
            y = model.fit_predict(X)
            view_projection_ax(projector, watch_params, X, ax, hue=y, title=name)
        else:
            view_projection_ax(projector, watch_params, X, ax, hue=hue)


def view_projection_ax(projector, watch_params, X, ax, hue, title=""):
    for k, v in projector.get_params().items():
        if k in watch_params:
            if title == "":
                title = "- ("
            title += f"{k}:{v} "
    if title != "":
        title = " " + title.strip() + ")"
    ax.set_title(type(projector).__name__ + title)

    X_out = projector.fit_transform(X)

    sns.scatterplot(x=X_out.T[0], y=X_out.T[1], hue=hue, ax=ax)


def view_clusters(models, projectors, X, p=1.0):
    n = int(X.shape[0] * p)

    X = X[:n]

    for model in tqdm(models):
        labels = model.fit_predict(X)

        view_projection(projectors, X, hue=labels)
