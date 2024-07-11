import os
import sys
import torch
import argparse
from math import degrees

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import rotation.losses as L
from ddn.ddn.pytorch.node import *
from rotation.nodes import RigitNodeConstraint, SVDLayer, IFTLayer
from rotation.datasets import get_dataset
from utils import get_initial_weights, plot_graphs_w, plot_graphs_loss

import warnings

warnings.filterwarnings("ignore")

# set main options
torch.set_printoptions(linewidth=200)
torch.set_printoptions(precision=4)


def run_optimization(optimization_type, P, Q, R_true, opt):
    print()
    print(optimization_type)

    # get the batch size and the number of points
    b, _, n = P.shape

    # set upper-level objective J
    if opt.upper_level == "geometric":
        J = L.angle_error
    elif opt.upper_level == "algebraic":
        J = L.frobenius_norm
    else:
        raise Exception("Upper-level loss is undefined.")

    # init weights
    w_init = get_initial_weights(b, n, opt.init_weights)
    w = w_init.clone().detach().requires_grad_()

    if optimization_type == "IFT":
        optimization_layer = IFTLayer()
    elif optimization_type == "DDN":
        node = RigitNodeConstraint()
        optimization_layer = DeclarativeLayer(node)
    elif optimization_type == "SVD":
        optimization_layer = SVDLayer
    else:
        raise Exception("Wrong optimization type.")

    # set optimizer
    optimizer = torch.optim.SGD([w], lr=opt.lr)

    loss_history = []
    w_history = [[w[0, 0].item()], [w[0, 1].item()], [w[0, 2].item()], [w[0, 3].item()]]

    def reevaluate():
        optimizer.zero_grad()
        R = optimization_layer(P, Q, w)
        loss = J(R_true, R)
        loss_history.append(loss[0].item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        return loss

    # optimize
    for i in range(opt.num_iter):
        optimizer.step(reevaluate)

        w_history[0].append(torch.nn.functional.relu(w[0, 0]).item())
        w_history[1].append(torch.nn.functional.relu(w[0, 1]).item())
        w_history[2].append(torch.nn.functional.relu(w[0, 2]).item())
        w_history[3].append(torch.nn.functional.relu(w[0, 3]).item())

    # get final results
    w = torch.nn.functional.relu(w)  # enforce non-negativity
    R = optimization_layer(P, Q, w)

    # compute errors
    angle_error = L.angle_error(R_true, R)
    frob_norm = L.frobenius_norm(R_true, R)

    # print errors
    print(
        "Rotation Error: {:0.4f} degrees".format(
            degrees(angle_error[0, ...].squeeze().detach().numpy())
        )
    )
    print("Algebraic Error: {}".format(frob_norm[0, ...].detach().numpy()))

    return w_history, loss_history


def main(opt):
    print("Rotation matrix estimation")

    # load dataset
    P, Q, R_true = get_dataset()

    w_history_svd = None
    w_history_ddn = None
    loss_history_svd = None
    loss_history_ddn = None

    # run optimization
    w_history_ift, loss_history_ift = run_optimization(
        "IFT", P, Q, R_true, opt
    )  # True by default
    if opt.autograd:
        w_history_svd, loss_history_svd = run_optimization("SVD", P, Q, R_true, opt)
    if opt.ddn:
        w_history_ddn, loss_history_ddn = run_optimization("DDN", P, Q, R_true, opt)

    # plot values of w and global loss
    if opt.plot:
        plot_graphs_w(
            w_history_ift,
            w_history_svd=w_history_svd,
            w_history_ddn=w_history_ddn,
            out_dir=os.path.join(opt.out, "rotation"),
        )

        plot_graphs_loss(
            loss_history_ift,
            loss_history_svd=loss_history_svd,
            loss_history_ddn=loss_history_ddn,
            out_dir=os.path.join(opt.out, "rotation"),
        )

    print()
    print("done")
    return 0


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Rotation matrix estimation")
    PARSER.add_argument(
        "--plot",
        action="store_true",
        help="plot inlier and outlier values, default False",
    )
    PARSER.add_argument(
        "--ift",
        action="store_true",
        help="compute R and w with backpropagation via IFT, default True",
    )
    PARSER.add_argument(
        "--ddn",
        action="store_true",
        help="compute R and w with backpropagation via DDN, default False",
    )
    PARSER.add_argument(
        "--autograd",
        action="store_true",
        help="compute R and w with backpropagation via autograd, default False",
    )
    PARSER.add_argument(
        "--init_weights",
        type=str,
        default="uniform",
        help="initialization for weights: uniform|random , default 'uniform'",
    )
    PARSER.add_argument(
        "--upper_level",
        type=str,
        default="geometric",
        help="upper-level objective: geometric|algebraic , default 'geometric'",
    )
    PARSER.add_argument(
        "--out",
        type=str,
        default="results",
        help="directory to save R and w and graphs",
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
        default=1e-1,
        help="learning rate, default 0.1",
    )

    ARGS = PARSER.parse_args()

    main(ARGS)
