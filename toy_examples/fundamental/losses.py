import torch


def frobenius_norm(F_true, F):
    """Squared Frobenius norm of matrices' difference"""
    return ((F - F_true) ** 2).sum(dim=(-2, -1))
