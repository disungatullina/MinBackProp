import os
import sys
import torch
import argparse
import warnings

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


from ddn.ddn.pytorch.node import *

import fundamental.losses as L
from fundamental.datasets import get_dataset
from fundamental.nodes import SVDLayer, FundamentalNodeConstraint, IFTLayer
from utils import get_initial_weights, plot_graphs_w, plot_graphs_loss

torch.set_printoptions(precision=4)

warnings.filterwarnings("ignore")


def run_optimization(optimization_type, A, B, F_true, opt):
    print(optimization_type)

    # get the batch size and the number of points
    b, n, _ = A.shape

    # set upper-level objective J
    J = L.frobenius_norm

    # init weights
    w_init = get_initial_weights(b, n, opt.init_weights)
    w = w_init.clone().detach().requires_grad_()

    if optimization_type == "IFT":
        optimization_layer = IFTLayer()
    elif optimization_type == "DDN":
        node = FundamentalNodeConstraint()
        optimization_layer = DeclarativeLayer(node)
    elif optimization_type == "SVD":
        optimization_layer = SVDLayer
    else:
        raise Exception("Wrong optimization type.")

    # set optimizer
    optimizer = torch.optim.SGD(
        [w],
        lr=opt.lr,
    )

    loss_history = []
    w_history = [[w[0, i].item()] for i in range(n)]

    def reevaluate():
        optimizer.zero_grad()
        F = optimization_layer(A, B, w)
        if (F * F_true).sum(dim=(-2, -1)) < 0:
            F_ = -F
        else:
            F_ = F
        loss = J(F_true, F_)
        loss_history.append(loss[0].item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        return loss

    for i in range(opt.num_iter):
        optimizer.step(reevaluate)

        w_ = torch.nn.functional.relu(w)
        w_ /= w_.sum(dim=1, keepdim=True)
        for i in range(n):
            w_history[i].append(torch.nn.functional.relu(w_[0, i]).item())

    # get final results
    w = torch.nn.functional.relu(w)
    w /= w.sum(dim=1, keepdim=True)
    F = optimization_layer(A, B, w)
    F = F / torch.norm(F)
    if (F * F_true).sum(dim=(-2, -1)) < 0:
        F = -F

    # compute error
    frob_norm = L.frobenius_norm(F_true, F)

    print("Algebraic Error: {}".format(frob_norm[0, ...].detach().numpy()))

    return w_history, loss_history


def main(opt):
    print("Fundamental matrix estimation")
    print()

    # load dataset
    A, B, F_true, w_true = get_dataset()

    w_history_svd = None
    w_history_ddn = None
    loss_history_svd = None
    loss_history_ddn = None

    # run optimization
    w_history_ift, loss_history_ift = run_optimization(
        "IFT", A, B, F_true, opt
    )  # True by default
    if opt.autograd:
        print()
        w_history_svd, loss_history_svd = run_optimization("SVD", A, B, F_true, opt)
    if opt.ddn:
        print()
        w_history_ddn, loss_history_ddn = run_optimization("DDN", A, B, F_true, opt)

    # plot values of w and global loss
    if opt.plot:
        plot_graphs_w(
            w_history_ift,
            w_history_svd=w_history_svd,
            w_history_ddn=w_history_ddn,
            out_dir=os.path.join(opt.out, "fundamental"),
        )

        plot_graphs_loss(
            loss_history_ift,
            loss_history_svd=loss_history_svd,
            loss_history_ddn=loss_history_ddn,
            out_dir=os.path.join(opt.out, "fundamental"),
        )

    print()
    print("done")
    return 0


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Fundamental matrix estimation")
    PARSER.add_argument(
        "--plot",
        action="store_true",
        help="plot inlier and outlier values, default False",
    )
    PARSER.add_argument(
        "--ift",
        action="store_true",
        help="compute F and w with backpropagation via IFT, default True",
    )
    PARSER.add_argument(
        "--ddn",
        action="store_true",
        help="compute F and w with backpropagation via DDN, default False",
    )
    PARSER.add_argument(
        "--autograd",
        action="store_true",
        help="compute F and w with backpropagation via autograd, default False",
    )
    PARSER.add_argument(
        "--init_weights",
        type=str,
        default="uniform",
        help="initialization for weights: uniform|random , default 'uniform'",
    )
    PARSER.add_argument(
        "--out",
        type=str,
        default="results",
        help="directory to save graphs",
    )
    PARSER.add_argument(
        "--num_iter",
        type=int,
        default=30,
        help="the number of iterations, default 30",
    )
    # PARSER.add_argument("-bs", "--batch_size", type=int, default=1, help="batch size")
    PARSER.add_argument(
        "--lr",
        type=float,
        default=1000.0,
        help="learning rate, default 1000.",
    )

    ARGS = PARSER.parse_args()

    main(ARGS)
