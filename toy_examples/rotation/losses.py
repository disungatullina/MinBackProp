import torch


def angle_error(A, B, w, R_true, y=None):
    """Pytorch inplementation of (0.5 * trace(R^(-1)R) - 1)"""
    max_dot_product = 1.0 - 1e-8
    error_rotation = (
        (0.5 * ((y * R_true).sum(dim=(-2, -1)) - 1.0))
        # .clamp_(-max_dot_product, max_dot_product)
        .clamp_(-max_dot_product, max_dot_product)
        .acos()
    )
    return error_rotation


def frobenius_norm(A, B, w, R_true, y=None):
    return ((y - R_true) ** 2).sum(dim=(-2, -1))
