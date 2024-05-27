import cv2
import torch
import argparse
import torch.nn as nn


def create_parser(description):
    """Create a default command line parser with the most common options.

    Keyword arguments:
    description -- description of the main functionality of a script/program
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,  # EF_hist_size_10_snn_0_85_thr_3.pth
        help="The name of the model to be used",
    )
    parser.add_argument(
        "--data_path",
        "-pth",
        default="dataset",  # EF_hist_size_10_snn_0_85_thr_3.pth
        help="The path you sed the dataset.",
    )
    parser.add_argument("--device", "-d", default="cuda", help="The device")
    parser.add_argument(
        "--detector",
        "-dt",
        default="rootsift",
        help="The detector used for obtaining local features. Values = loftr, sift",
    )
    parser.add_argument(
        "--snn", "-snn", default=0.80, help="The SNN ratio threshold for SIFT."
    )
    parser.add_argument(
        "--nfeatures",
        "-nf",
        type=int,
        default=2000,
        help="The expected number of features in SIFT.",
    )
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="batch size")
    parser.add_argument(
        "--ransac_batch_size", "-rbs", type=int, default=64, help="ransac batch size"
    )
    parser.add_argument(
        "--scoring",
        "-s",
        type=int,
        default=1,
        help="The used scoring function. 0 - RANSAC, 1 - MSAC",
    )
    parser.add_argument(
        "--precision",
        "-pr",
        type=int,
        default=1,
        help="The used data precision. 0 - float16, 1 - float32, 2-float64",
    )
    parser.add_argument("--tr", "-tr", type=int, default=0, help="1 - train, 0v- test")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.75,
        help="Inlier-outlier threshold. "
        "It will be normalized for E matrix estimation inside the code using focal length.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="Epochs for training. "
        "It will be the epoch number used  inside training.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-4,
        help="learning rate for network optimizer.",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        type=int,
        default=0,
        help="how many workers for data loader",
    )
    parser.add_argument(
        "--datasets",
        "-ds",
        default="st_peters_square",
        help="the datasets we would like to use",
    )
    parser.add_argument(
        "--batch_mode", "-bm", type=int, default=0, help="use the provided data list"
    )
    parser.add_argument(
        "--prob",
        "-p",
        type=int,
        default=2,
        help="the way we use the weights, 0-normalized weights, 1-unnormarlized weights, 2-logits",
    )
    parser.add_argument(
        "--session",
        "-sid",
        default="",
        help="custom session name appended to output files, "
        "useful to separate different runs of a script",
    )
    parser.add_argument(
        "--topk",
        "-topk",
        default=False,
        help="use the errors of the best k models as the loss, otherwise, taaake the average.",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=300,
        help="the number of the best models included in the loss.",
    )
    parser.add_argument(
        "--ift",
        "-ift",
        type=int,
        default=1,
        help="the method to backpropagate; 0-autograd, 1-ift, 2-ddn; 1 is default",
    )

    return parser


def init_weights(m):
    """Customize the weight initialization process as ResNet does.

    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L208
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def create_session_string(prefix, epochs, nfeatures, ratio, session, threshold):
    """Create an identifier string from the most common parameter options.

    Keyword arguments:
    prefix -- custom string appended at the beginning of the session string
    epochs -- how many epochs you trained
    orb -- bool indicating whether ORB features or SIFT features are used
    rootsift -- bool indicating whether RootSIFT normalization is used
    ratio -- threshold for Lowe's ratio filter
    session -- custom string appended at the end of the session string
    """
    session_string = prefix + "_"
    session_string += "E_"
    session_string += "e_" + str(epochs) + "_"
    # if rootsift: session_string += 'rs_'
    session_string += "rs_" + str(nfeatures)
    session_string += "_r%.2f_" % ratio
    session_string += "t%.2f_" % threshold
    # specific id if we train the same config for times
    session_string += session

    return session_string


outdoor_test_datasets = [
    "buckingham_palace",
    "brandenburg_gate",
    "colosseum_exterior",
    "grand_place_brussels",
    "notre_dame_front_facade",
    "palace_of_westminster",
    "pantheon_exterior",
    "prague_old_town_square",
    "sacre_coeur",
    "taj_mahal",
    "trevi_fountain",
    "westminster_abbey",
]


test_datasets = outdoor_test_datasets
