import os
import torch
import matplotlib.pyplot as plt


def get_initial_weights(b, n, init_type):
    # Generate weights (uniform):
    if init_type == "uniform":
        w = torch.ones(b, n, dtype=torch.float)  # b x n
        w = w.div(w.sum(-1).unsqueeze(-1))
    # Generate weights (random):
    elif init_type == "random":
        w = torch.rand(b, n, dtype=torch.float)
    else:
        w = None
    return w


def plot_graphs_w(
    w_history_ift, w_history_svd=None, w_history_ddn=None, out_dir="results"
):
    plt.rcParams["font.size"] = 14
    for i in range(len(w_history_ift)):
        if i == 0:
            Y_LABEL = "weight of the outlier"
        else:
            Y_LABEL = "weight of an inlier"
        plt.figure(figsize=(9, 6))
        plt.axes()
        if w_history_svd is not None:
            plt.plot(w_history_svd[i], label="SVD", color="red", linewidth=4)
        if w_history_ddn is not None:
            plt.plot(w_history_ddn[i], "--", label="DDN", linewidth=4)
        plt.plot(w_history_ift[i], "--", label="IFT", color="green", linewidth=4)
        plt.ylabel(
            Y_LABEL,
            fontsize=18,
        )
        plt.xlabel(
            "# iterations",
            fontsize=18,
        )
        plt.xticks(
            fontsize=18,
        )
        plt.yticks(fontsize=18)
        plt.legend(loc="lower right")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, "w_{}.png".format(i)))
        plt.show()


def plot_graphs_loss(
    loss_history_ift, loss_history_svd=None, loss_history_ddn=None, out_dir="results"
):
    plt.figure(figsize=(9, 6))
    if loss_history_svd is not None:
        plt.plot(loss_history_svd, label="SVD", color="red", linewidth=4)
    if loss_history_ddn is not None:
        plt.plot(loss_history_ddn, "--", label="DDN", linewidth=4)
    plt.plot(loss_history_ift, "--", label="IFT", color="green", linewidth=4)
    plt.ylabel(
        "global loss",
        fontsize=18,
    )
    plt.xlabel(
        "# iterations",
        fontsize=18,
    )
    plt.legend(loc="upper right")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, "global_loss.png"))
