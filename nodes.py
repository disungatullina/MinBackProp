import math
import torch
import random
import time

import numpy as np
import torch.nn as nn

from torch.autograd import Function

from ddn.ddn.pytorch.node import *

import estimators.essential_matrix_estimator_nister as ns


############ IFT function ############


class IFTFunction(torch.autograd.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, minimal_samples, E_true, estimator):
        """
        E_true: 3 x 3
        minimal_samples : b x 5 x 4
        """
        est = estimator.estimate_model(minimal_samples)

        solution_num = 10

        distances = torch.norm(
            est - E_true.unsqueeze(0).repeat(est.shape[0], 1, 1), dim=(1, 2)
        ).view(est.shape[0], -1)

        try:
            chosen_indices = torch.argmin(distances.view(-1, solution_num), dim=-1)
            chosen_models = torch.stack(
                [
                    (est.view(-1, solution_num, 3, 3))[i, chosen_indices[i], :]
                    for i in range(int(est.shape[0] / solution_num))
                ]
            )

        except ValueError as e:
            print(
                "not enough models for selection, we choose the first solution in this batch",
                e,
                est.shape,
            )
            chosen_models = est[0].unsqueeze(0)

        ctx.save_for_backward(minimal_samples, chosen_models)

        return chosen_models

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        minimal_samples, _, _ = inputs
        ctx.save_for_backward(minimal_samples, output)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        minimal_samples, output = ctx.saved_tensors  # output -- [b, 3, 3]
        grad_E_true = grad_minimal_samples = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        b = grad_output.shape[0]

        # E - bx3x3 ; minimal_sample - bx5x4 ; J_Ex - bx9x20 ;  grad_output - bx3x3
        J_Ex = compute_jacobians(output, minimal_samples)  # b x 9 x 20
        J_Ex = torch.einsum("bi,bij->bj", grad_output.view(b, 9), J_Ex)
        grad_minimal_samples = J_Ex.view(b, 5, 4)

        return grad_minimal_samples, None, None


class IFTLayer(nn.Module):
    def __init__(self, device, ift):
        super().__init__()
        self.estimator = ns.EssentialMatrixEstimatorNister(device=device, ift=ift)

    def forward(self, minimal_samples, E_true):
        return IFTFunction.apply(minimal_samples, E_true, self.estimator)


def compute_jacobians(E, minimal_sample):
    """
    E -- b x 3 x 3
    minimal_sample -- b x 5 x 4
    """
    b = E.shape[0]

    E11 = E[:, 0, 0]
    E12 = E[:, 0, 1]
    E13 = E[:, 0, 2]
    E21 = E[:, 1, 0]
    E22 = E[:, 1, 1]
    E23 = E[:, 1, 2]
    E31 = E[:, 2, 0]
    E32 = E[:, 2, 1]
    E33 = E[:, 2, 2]

    x11 = minimal_sample[:, 0, 0]
    y11 = minimal_sample[:, 0, 1]

    x12 = minimal_sample[:, 0, 2]
    y12 = minimal_sample[:, 0, 3]

    x21 = minimal_sample[:, 1, 0]
    y21 = minimal_sample[:, 1, 1]

    x22 = minimal_sample[:, 1, 2]
    y22 = minimal_sample[:, 1, 3]

    x31 = minimal_sample[:, 2, 0]
    y31 = minimal_sample[:, 2, 1]

    x32 = minimal_sample[:, 2, 2]
    y32 = minimal_sample[:, 2, 3]

    x41 = minimal_sample[:, 3, 0]
    y41 = minimal_sample[:, 3, 1]

    x42 = minimal_sample[:, 3, 2]
    y42 = minimal_sample[:, 3, 3]

    x51 = minimal_sample[:, 4, 0]
    y51 = minimal_sample[:, 4, 1]

    x52 = minimal_sample[:, 4, 2]
    y52 = minimal_sample[:, 4, 3]

    J_E = torch.zeros((b, 16, 9), device=minimal_sample.device)

    J_E[:, 0, 0] = x12 * x11
    J_E[:, 1, 0] = x22 * x21
    J_E[:, 2, 0] = x32 * x31
    J_E[:, 3, 0] = x42 * x41
    J_E[:, 4, 0] = x52 * x51
    J_E[:, 5, 0] = 2 * E11
    J_E[:, 6, 0] = (
        3 * E11**2
        + E12**2
        + E13**2
        + E21**2
        - E22**2
        - E23**2
        + E31**2
        - E32**2
        - E33**2
    )
    J_E[:, 7, 0] = 2 * E11 * E12 + 2 * E21 * E22 + 2 * E31 * E32
    J_E[:, 8, 0] = 2 * E11 * E13 + 2 * E21 * E23 + 2 * E31 * E33
    J_E[:, 9, 0] = 2 * E11 * E21 + 2 * E12 * E22 + 2 * E13 * E23
    J_E[:, 10, 0] = -2 * E11 * E22 + 2 * E12 * E21
    J_E[:, 11, 0] = -2 * E11 * E23 + 2 * E13 * E21
    J_E[:, 12, 0] = 2 * E11 * E31 + 2 * E12 * E32 + 2 * E13 * E33
    J_E[:, 13, 0] = -2 * E11 * E32 + 2 * E12 * E31
    J_E[:, 14, 0] = -2 * E11 * E33 + 2 * E13 * E31
    J_E[:, 15, 0] = E22 * E33 - E23 * E32

    J_E[:, 0, 1] = x12 * y11
    J_E[:, 1, 1] = x22 * y21
    J_E[:, 2, 1] = x32 * y31
    J_E[:, 3, 1] = x42 * y41
    J_E[:, 4, 1] = x52 * y51
    J_E[:, 5, 1] = 2 * E12
    J_E[:, 6, 1] = 2 * E11 * E12 + 2 * E21 * E22 + 2 * E31 * E32
    J_E[:, 7, 1] = (
        E11**2
        + 3 * E12**2
        + E13**2
        - E21**2
        + E22**2
        - E23**2
        - E31**2
        + E32**2
        - E33**2
    )
    J_E[:, 8, 1] = 2 * E12 * E13 + 2 * E22 * E23 + 2 * E32 * E33
    J_E[:, 9, 1] = 2 * E11 * E22 - 2 * E12 * E21
    J_E[:, 10, 1] = 2 * E11 * E21 + 2 * E12 * E22 + 2 * E13 * E23
    J_E[:, 11, 1] = -2 * E12 * E23 + 2 * E13 * E22
    J_E[:, 12, 1] = 2 * E11 * E32 - 2 * E12 * E31
    J_E[:, 13, 1] = 2 * E11 * E31 + 2 * E12 * E32 + 2 * E13 * E33
    J_E[:, 14, 1] = -2 * E12 * E33 + 2 * E13 * E32
    J_E[:, 15, 1] = -E21 * E33 + E23 * E31

    J_E[:, 0, 2] = x12
    J_E[:, 1, 2] = x22
    J_E[:, 2, 2] = x32
    J_E[:, 3, 2] = x42
    J_E[:, 4, 2] = x52
    J_E[:, 5, 2] = 2 * E13
    J_E[:, 6, 2] = 2 * E11 * E13 + 2 * E21 * E23 + 2 * E31 * E33
    J_E[:, 7, 2] = 2 * E12 * E13 + 2 * E22 * E23 + 2 * E32 * E33
    J_E[:, 8, 2] = (
        E11**2
        + E12**2
        + 3 * E13**2
        - E21**2
        - E22**2
        + E23**2
        - E31**2
        - E32**2
        + E33**2
    )
    J_E[:, 9, 2] = 2 * E11 * E23 - 2 * E13 * E21
    J_E[:, 10, 2] = 2 * E12 * E23 - 2 * E13 * E22
    J_E[:, 11, 2] = 2 * E11 * E21 + 2 * E12 * E22 + 2 * E13 * E23
    J_E[:, 12, 2] = 2 * E11 * E33 - 2 * E13 * E31
    J_E[:, 13, 2] = 2 * E12 * E33 - 2 * E13 * E32
    J_E[:, 14, 2] = 2 * E11 * E31 + 2 * E12 * E32 + 2 * E13 * E33
    J_E[:, 15, 2] = E21 * E32 - E22 * E31

    J_E[:, 0, 3] = y12 * y11
    J_E[:, 1, 3] = y22 * y21
    J_E[:, 2, 3] = y32 * y31
    J_E[:, 3, 3] = y42 * y41
    J_E[:, 4, 3] = y52 * y51
    J_E[:, 5, 3] = 2 * E21
    J_E[:, 6, 3] = 2 * E11 * E21 + 2 * E12 * E22 + 2 * E13 * E23
    J_E[:, 7, 3] = 2 * E11 * E22 - 2 * E12 * E21
    J_E[:, 8, 3] = 2 * E11 * E23 - 2 * E13 * E21
    J_E[:, 9, 3] = (
        E11**2
        - E12**2
        - E13**2
        + 3 * E21**2
        + E22**2
        + E23**2
        + E31**2
        - E32**2
        - E33**2
    )
    J_E[:, 10, 3] = 2 * E11 * E12 + 2 * E21 * E22 + 2 * E31 * E32
    J_E[:, 11, 3] = 2 * E11 * E13 + 2 * E21 * E23 + 2 * E31 * E33
    J_E[:, 12, 3] = 2 * E21 * E31 + 2 * E22 * E32 + 2 * E23 * E33
    J_E[:, 13, 3] = -2 * E21 * E32 + 2 * E22 * E31
    J_E[:, 14, 3] = -2 * E21 * E33 + 2 * E23 * E31
    J_E[:, 15, 3] = -E12 * E33 + E13 * E32

    J_E[:, 0, 4] = y12 * y11
    J_E[:, 1, 4] = y22 * y21
    J_E[:, 2, 4] = y32 * y31
    J_E[:, 3, 4] = y42 * y41
    J_E[:, 4, 4] = y52 * y51
    J_E[:, 5, 4] = 2 * E22
    J_E[:, 6, 4] = -2 * E11 * E22 + 2 * E12 * E21
    J_E[:, 7, 4] = 2 * E11 * E21 + 2 * E12 * E22 + 2 * E13 * E23
    J_E[:, 8, 4] = 2 * E12 * E23 - 2 * E13 * E22
    J_E[:, 9, 4] = 2 * E11 * E12 + 2 * E21 * E22 + 2 * E31 * E32
    J_E[:, 10, 4] = (
        -(E11**2)
        + E12**2
        - E13**2
        + E21**2
        + 3 * E22**2
        + E23**2
        - E31**2
        + E32**2
        - E33**2
    )
    J_E[:, 11, 4] = 2 * E12 * E13 + 2 * E22 * E23 + 2 * E32 * E33
    J_E[:, 12, 4] = 2 * E21 * E32 - 2 * E22 * E31
    J_E[:, 13, 4] = 2 * E21 * E31 + 2 * E22 * E32 + 2 * E23 * E33
    J_E[:, 14, 4] = -2 * E22 * E33 + 2 * E23 * E32
    J_E[:, 15, 4] = E11 * E33 - E13 * E31

    J_E[:, 0, 5] = y12
    J_E[:, 1, 5] = y22
    J_E[:, 2, 5] = y32
    J_E[:, 3, 5] = y42
    J_E[:, 4, 5] = y52
    J_E[:, 5, 5] = 2 * E23
    J_E[:, 6, 5] = -2 * E11 * E23 + 2 * E13 * E21
    J_E[:, 7, 5] = -2 * E12 * E23 + 2 * E13 * E22
    J_E[:, 8, 5] = 2 * E11 * E21 + 2 * E12 * E22 + 2 * E13 * E23
    J_E[:, 9, 5] = 2 * E11 * E13 + 2 * E21 * E23 + 2 * E31 * E33
    J_E[:, 10, 5] = 2 * E12 * E13 + 2 * E22 * E23 + 2 * E32 * E33
    J_E[:, 11, 5] = (
        -(E11**2)
        - E12**2
        + E13**2
        + E21**2
        + E22**2
        + 3 * E23**2
        - E31**2
        - E32**2
        + E33**2
    )
    J_E[:, 12, 5] = 2 * E21 * E33 - 2 * E23 * E31
    J_E[:, 13, 5] = 2 * E22 * E33 - 2 * E23 * E32
    J_E[:, 14, 5] = 2 * E21 * E31 + 2 * E22 * E32 + 2 * E23 * E33
    J_E[:, 15, 5] = -E11 * E32 + E12 * E31

    J_E[:, 0, 6] = x11
    J_E[:, 1, 6] = x21
    J_E[:, 2, 6] = x31
    J_E[:, 3, 6] = x41
    J_E[:, 4, 6] = x51
    J_E[:, 5, 6] = 2 * E31
    J_E[:, 6, 6] = 2 * E11 * E31 + 2 * E12 * E32 + 2 * E13 * E33
    J_E[:, 7, 6] = 2 * E11 * E32 - 2 * E12 * E31
    J_E[:, 8, 6] = 2 * E11 * E33 - 2 * E13 * E31
    J_E[:, 9, 6] = 2 * E21 * E31 + 2 * E22 * E32 + 2 * E23 * E33
    J_E[:, 10, 6] = 2 * E21 * E32 - 2 * E22 * E31
    J_E[:, 11, 6] = 2 * E21 * E33 - 2 * E23 * E31
    J_E[:, 12, 6] = (
        E11**2
        - E12**2
        - E13**2
        + E21**2
        - E22**2
        - E23**2
        + 3 * E31**2
        + E32**2
        + E33**2
    )
    J_E[:, 13, 6] = 2 * E11 * E12 + 2 * E21 * E22 + 2 * E31 * E32
    J_E[:, 14, 6] = 2 * E11 * E13 + 2 * E21 * E23 + 2 * E31 * E33
    J_E[:, 15, 6] = E12 * E23 - E13 * E22

    J_E[:, 0, 7] = y11
    J_E[:, 1, 7] = y21
    J_E[:, 2, 7] = y31
    J_E[:, 3, 7] = y41
    J_E[:, 4, 7] = y51
    J_E[:, 5, 7] = 2 * E32
    J_E[:, 6, 7] = -2 * E11 * E32 + 2 * E12 * E31
    J_E[:, 7, 7] = 2 * E11 * E31 + 2 * E12 * E32 + 2 * E13 * E33
    J_E[:, 8, 7] = 2 * E12 * E33 - 2 * E13 * E32
    J_E[:, 9, 7] = -2 * E21 * E32 + 2 * E22 * E31
    J_E[:, 10, 7] = 2 * E21 * E31 + 2 * E22 * E32 + 2 * E23 * E33
    J_E[:, 11, 7] = 2 * E22 * E33 - 2 * E23 * E32
    J_E[:, 12, 7] = 2 * E11 * E12 + 2 * E21 * E22 + 2 * E31 * E32
    J_E[:, 13, 7] = (
        -(E11**2)
        + E12**2
        - E13**2
        - E21**2
        + E22**2
        - E23**2
        + E31**2
        + 3 * E32**2
        + E33**2
    )
    J_E[:, 14, 7] = 2 * E12 * E13 + 2 * E22 * E23 + 2 * E32 * E33
    J_E[:, 15, 7] = -E11 * E23 + E13 * E21

    J_E[:, 0, 8] = 1
    J_E[:, 1, 8] = 1
    J_E[:, 2, 8] = 1
    J_E[:, 3, 8] = 1
    J_E[:, 4, 8] = 1
    J_E[:, 5, 8] = 2 * E33
    J_E[:, 6, 8] = -2 * E11 * E33 + 2 * E13 * E31
    J_E[:, 7, 8] = -2 * E12 * E33 + 2 * E13 * E32
    J_E[:, 8, 8] = 2 * E11 * E31 + 2 * E12 * E32 + 2 * E13 * E33
    J_E[:, 9, 8] = -2 * E21 * E33 + 2 * E23 * E31
    J_E[:, 10, 8] = -2 * E22 * E33 + 2 * E23 * E32
    J_E[:, 11, 8] = 2 * E21 * E31 + 2 * E22 * E32 + 2 * E23 * E33
    J_E[:, 12, 8] = 2 * E11 * E13 + 2 * E21 * E23 + 2 * E31 * E33
    J_E[:, 13, 8] = 2 * E12 * E13 + 2 * E22 * E23 + 2 * E32 * E33
    J_E[:, 14, 8] = (
        -(E11**2)
        - E12**2
        + E13**2
        - E21**2
        - E22**2
        + E23**2
        + E31**2
        + E32**2
        + 3 * E33**2
    )
    J_E[:, 15, 8] = E11 * E22 - E12 * E21

    J_x = torch.zeros((b, 16, 20), device=minimal_sample.device)

    J_x[:, 0, 0] = E11 * x12 + E21 * y12 + E31
    J_x[:, 1, 0] = 0
    J_x[:, 2, 0] = 0
    J_x[:, 3, 0] = 0
    J_x[:, 4, 0] = 0
    J_x[:, 5, 0] = 0
    J_x[:, 6, 0] = 0
    J_x[:, 7, 0] = 0
    J_x[:, 8, 0] = 0
    J_x[:, 9, 0] = 0
    J_x[:, 10, 0] = 0
    J_x[:, 11, 0] = 0
    J_x[:, 12, 0] = 0
    J_x[:, 13, 0] = 0
    J_x[:, 14, 0] = 0
    J_x[:, 15, 0] = 0

    J_x[:, 0, 1] = E12 * x12 + E22 * y12 + E32
    J_x[:, 1, 1] = 0
    J_x[:, 2, 1] = 0
    J_x[:, 3, 1] = 0
    J_x[:, 4, 1] = 0
    J_x[:, 5, 1] = 0
    J_x[:, 6, 1] = 0
    J_x[:, 7, 1] = 0
    J_x[:, 8, 1] = 0
    J_x[:, 9, 1] = 0
    J_x[:, 10, 1] = 0
    J_x[:, 11, 1] = 0
    J_x[:, 12, 1] = 0
    J_x[:, 13, 1] = 0
    J_x[:, 14, 1] = 0
    J_x[:, 15, 1] = 0

    J_x[:, 0, 2] = E11 * x11 + E12 * y11 + E13
    J_x[:, 1, 2] = 0
    J_x[:, 2, 2] = 0
    J_x[:, 3, 2] = 0
    J_x[:, 4, 2] = 0
    J_x[:, 5, 2] = 0
    J_x[:, 6, 2] = 0
    J_x[:, 7, 2] = 0
    J_x[:, 8, 2] = 0
    J_x[:, 9, 2] = 0
    J_x[:, 10, 2] = 0
    J_x[:, 11, 2] = 0
    J_x[:, 12, 2] = 0
    J_x[:, 13, 2] = 0
    J_x[:, 14, 2] = 0
    J_x[:, 15, 2] = 0

    J_x[:, 0, 3] = E21 * x11 + E22 * y11 + E23
    J_x[:, 1, 3] = 0
    J_x[:, 2, 3] = 0
    J_x[:, 3, 3] = 0
    J_x[:, 4, 3] = 0
    J_x[:, 5, 3] = 0
    J_x[:, 6, 3] = 0
    J_x[:, 7, 3] = 0
    J_x[:, 8, 3] = 0
    J_x[:, 9, 3] = 0
    J_x[:, 10, 3] = 0
    J_x[:, 11, 3] = 0
    J_x[:, 12, 3] = 0
    J_x[:, 13, 3] = 0
    J_x[:, 14, 3] = 0
    J_x[:, 15, 3] = 0

    J_x[:, 0, 4] = 0
    J_x[:, 1, 4] = E11 * x22 + E21 * y22 + E31
    J_x[:, 2, 4] = 0
    J_x[:, 3, 4] = 0
    J_x[:, 4, 4] = 0
    J_x[:, 5, 4] = 0
    J_x[:, 6, 4] = 0
    J_x[:, 7, 4] = 0
    J_x[:, 8, 4] = 0
    J_x[:, 9, 4] = 0
    J_x[:, 10, 4] = 0
    J_x[:, 11, 4] = 0
    J_x[:, 12, 4] = 0
    J_x[:, 13, 4] = 0
    J_x[:, 14, 4] = 0
    J_x[:, 15, 4] = 0

    J_x[:, 0, 5] = 0
    J_x[:, 1, 5] = E12 * x22 + E22 * y22 + E32
    J_x[:, 2, 5] = 0
    J_x[:, 3, 5] = 0
    J_x[:, 4, 5] = 0
    J_x[:, 5, 5] = 0
    J_x[:, 6, 5] = 0
    J_x[:, 7, 5] = 0
    J_x[:, 8, 5] = 0
    J_x[:, 9, 5] = 0
    J_x[:, 10, 5] = 0
    J_x[:, 11, 5] = 0
    J_x[:, 12, 5] = 0
    J_x[:, 13, 5] = 0
    J_x[:, 14, 5] = 0
    J_x[:, 15, 5] = 0

    J_x[:, 0, 6] = 0
    J_x[:, 1, 6] = E11 * x21 + E12 * y21 + E13
    J_x[:, 2, 6] = 0
    J_x[:, 3, 6] = 0
    J_x[:, 4, 6] = 0
    J_x[:, 5, 6] = 0
    J_x[:, 6, 6] = 0
    J_x[:, 7, 6] = 0
    J_x[:, 8, 6] = 0
    J_x[:, 9, 6] = 0
    J_x[:, 10, 6] = 0
    J_x[:, 11, 6] = 0
    J_x[:, 12, 6] = 0
    J_x[:, 13, 6] = 0
    J_x[:, 14, 6] = 0
    J_x[:, 15, 6] = 0

    J_x[:, 0, 7] = 0
    J_x[:, 1, 7] = E21 * x21 + E22 * y21 + E23
    J_x[:, 2, 7] = 0
    J_x[:, 3, 7] = 0
    J_x[:, 4, 7] = 0
    J_x[:, 5, 7] = 0
    J_x[:, 6, 7] = 0
    J_x[:, 7, 7] = 0
    J_x[:, 8, 7] = 0
    J_x[:, 9, 7] = 0
    J_x[:, 10, 7] = 0
    J_x[:, 11, 7] = 0
    J_x[:, 12, 7] = 0
    J_x[:, 13, 7] = 0
    J_x[:, 14, 7] = 0
    J_x[:, 15, 7] = 0

    J_x[:, 0, 8] = 0
    J_x[:, 1, 8] = 0
    J_x[:, 2, 8] = E11 * x32 + E21 * y32 + E31
    J_x[:, 3, 8] = 0
    J_x[:, 4, 8] = 0
    J_x[:, 5, 8] = 0
    J_x[:, 6, 8] = 0
    J_x[:, 7, 8] = 0
    J_x[:, 8, 8] = 0
    J_x[:, 9, 8] = 0
    J_x[:, 10, 8] = 0
    J_x[:, 11, 8] = 0
    J_x[:, 12, 8] = 0
    J_x[:, 13, 8] = 0
    J_x[:, 14, 8] = 0
    J_x[:, 15, 8] = 0

    J_x[:, 0, 9] = 0
    J_x[:, 1, 9] = 0
    J_x[:, 2, 9] = E12 * x32 + E22 * y32 + E32
    J_x[:, 3, 9] = 0
    J_x[:, 4, 9] = 0
    J_x[:, 5, 9] = 0
    J_x[:, 6, 9] = 0
    J_x[:, 7, 9] = 0
    J_x[:, 8, 9] = 0
    J_x[:, 9, 9] = 0
    J_x[:, 10, 9] = 0
    J_x[:, 11, 9] = 0
    J_x[:, 12, 9] = 0
    J_x[:, 13, 9] = 0
    J_x[:, 14, 9] = 0
    J_x[:, 15, 9] = 0

    J_x[:, 0, 10] = 0
    J_x[:, 1, 10] = 0
    J_x[:, 2, 10] = E11 * x31 + E12 * y31 + E13
    J_x[:, 3, 10] = 0
    J_x[:, 4, 10] = 0
    J_x[:, 5, 10] = 0
    J_x[:, 6, 10] = 0
    J_x[:, 7, 10] = 0
    J_x[:, 8, 10] = 0
    J_x[:, 9, 10] = 0
    J_x[:, 10, 10] = 0
    J_x[:, 11, 10] = 0
    J_x[:, 12, 10] = 0
    J_x[:, 13, 10] = 0
    J_x[:, 14, 10] = 0
    J_x[:, 15, 10] = 0

    J_x[:, 0, 11] = 0
    J_x[:, 1, 11] = 0
    J_x[:, 2, 11] = E21 * x31 + E22 * y31 + E23
    J_x[:, 3, 11] = 0
    J_x[:, 4, 11] = 0
    J_x[:, 5, 11] = 0
    J_x[:, 6, 11] = 0
    J_x[:, 7, 11] = 0
    J_x[:, 8, 11] = 0
    J_x[:, 9, 11] = 0
    J_x[:, 10, 11] = 0
    J_x[:, 11, 11] = 0
    J_x[:, 12, 11] = 0
    J_x[:, 13, 11] = 0
    J_x[:, 14, 11] = 0
    J_x[:, 15, 11] = 0

    J_x[:, 0, 12] = 0
    J_x[:, 1, 12] = 0
    J_x[:, 2, 12] = 0
    J_x[:, 3, 12] = E11 * x42 + E21 * y42 + E31
    J_x[:, 4, 12] = 0
    J_x[:, 5, 12] = 0
    J_x[:, 6, 12] = 0
    J_x[:, 7, 12] = 0
    J_x[:, 8, 12] = 0
    J_x[:, 9, 12] = 0
    J_x[:, 10, 12] = 0
    J_x[:, 11, 12] = 0
    J_x[:, 12, 12] = 0
    J_x[:, 13, 12] = 0
    J_x[:, 14, 12] = 0
    J_x[:, 15, 12] = 0

    J_x[:, 0, 13] = 0
    J_x[:, 1, 13] = 0
    J_x[:, 2, 13] = 0
    J_x[:, 3, 13] = E12 * x42 + E22 * y42 + E32
    J_x[:, 4, 13] = 0
    J_x[:, 5, 13] = 0
    J_x[:, 6, 13] = 0
    J_x[:, 7, 13] = 0
    J_x[:, 8, 13] = 0
    J_x[:, 9, 13] = 0
    J_x[:, 10, 13] = 0
    J_x[:, 11, 13] = 0
    J_x[:, 12, 13] = 0
    J_x[:, 13, 13] = 0
    J_x[:, 14, 13] = 0
    J_x[:, 15, 13] = 0

    J_x[:, 0, 14] = 0
    J_x[:, 1, 14] = 0
    J_x[:, 2, 14] = 0
    J_x[:, 3, 14] = E11 * x41 + E12 * y41 + E13
    J_x[:, 4, 14] = 0
    J_x[:, 5, 14] = 0
    J_x[:, 6, 14] = 0
    J_x[:, 7, 14] = 0
    J_x[:, 8, 14] = 0
    J_x[:, 9, 14] = 0
    J_x[:, 10, 14] = 0
    J_x[:, 11, 14] = 0
    J_x[:, 12, 14] = 0
    J_x[:, 13, 14] = 0
    J_x[:, 14, 14] = 0
    J_x[:, 15, 14] = 0

    J_x[:, 0, 15] = 0
    J_x[:, 1, 15] = 0
    J_x[:, 2, 15] = 0
    J_x[:, 3, 15] = E21 * x41 + E22 * y41 + E23
    J_x[:, 4, 15] = 0
    J_x[:, 5, 15] = 0
    J_x[:, 6, 15] = 0
    J_x[:, 7, 15] = 0
    J_x[:, 8, 15] = 0
    J_x[:, 9, 15] = 0
    J_x[:, 10, 15] = 0
    J_x[:, 11, 15] = 0
    J_x[:, 12, 15] = 0
    J_x[:, 13, 15] = 0
    J_x[:, 14, 15] = 0
    J_x[:, 15, 15] = 0

    J_x[:, 0, 16] = 0
    J_x[:, 1, 16] = 0
    J_x[:, 2, 16] = 0
    J_x[:, 3, 16] = 0
    J_x[:, 4, 16] = E11 * x52 + E21 * y52 + E31
    J_x[:, 5, 16] = 0
    J_x[:, 6, 16] = 0
    J_x[:, 7, 16] = 0
    J_x[:, 8, 16] = 0
    J_x[:, 9, 16] = 0
    J_x[:, 10, 16] = 0
    J_x[:, 11, 16] = 0
    J_x[:, 12, 16] = 0
    J_x[:, 13, 16] = 0
    J_x[:, 14, 16] = 0
    J_x[:, 15, 16] = 0

    J_x[:, 0, 17] = 0
    J_x[:, 1, 17] = 0
    J_x[:, 2, 17] = 0
    J_x[:, 3, 17] = 0
    J_x[:, 4, 17] = E12 * x52 + E22 * y52 + E32
    J_x[:, 5, 17] = 0
    J_x[:, 6, 17] = 0
    J_x[:, 7, 17] = 0
    J_x[:, 8, 17] = 0
    J_x[:, 9, 17] = 0
    J_x[:, 10, 17] = 0
    J_x[:, 11, 17] = 0
    J_x[:, 12, 17] = 0
    J_x[:, 13, 17] = 0
    J_x[:, 14, 17] = 0
    J_x[:, 15, 17] = 0

    J_x[:, 0, 18] = 0
    J_x[:, 1, 18] = 0
    J_x[:, 2, 18] = 0
    J_x[:, 3, 18] = 0
    J_x[:, 4, 18] = E11 * x51 + E12 * y51 + E13
    J_x[:, 5, 18] = 0
    J_x[:, 6, 18] = 0
    J_x[:, 7, 18] = 0
    J_x[:, 8, 18] = 0
    J_x[:, 9, 18] = 0
    J_x[:, 10, 18] = 0
    J_x[:, 11, 18] = 0
    J_x[:, 12, 18] = 0
    J_x[:, 13, 18] = 0
    J_x[:, 14, 18] = 0
    J_x[:, 15, 18] = 0

    J_x[:, 0, 19] = 0
    J_x[:, 1, 19] = 0
    J_x[:, 2, 19] = 0
    J_x[:, 3, 19] = 0
    J_x[:, 4, 19] = E21 * x51 + E22 * y51 + E23
    J_x[:, 5, 19] = 0
    J_x[:, 6, 19] = 0
    J_x[:, 7, 19] = 0
    J_x[:, 8, 19] = 0
    J_x[:, 9, 19] = 0
    J_x[:, 10, 19] = 0
    J_x[:, 11, 19] = 0
    J_x[:, 12, 19] = 0
    J_x[:, 13, 19] = 0
    J_x[:, 14, 19] = 0
    J_x[:, 15, 19] = 0

    J_E9 = torch.zeros((b, 9, 9), device=minimal_sample.device)
    J_x9 = torch.zeros((b, 9, 20), device=minimal_sample.device)
    J_Ex = torch.zeros((b, 9, 20), device=minimal_sample.device)

    J_E9[:, :6, :] = J_E[:, :6, :]
    J_x9[:, :6, :] = J_x[:, :6, :]

    rows = []
    for i in range(3):
        rows.append(random.sample(range(7, 15), 3))

    J_E9[:, 6, :] = (
        J_E[:, rows[0][0], :] + J_E[:, rows[0][1], :] + J_E[:, rows[0][2], :]
    )
    J_E9[:, 7, :] = (
        J_E[:, rows[1][0], :] + J_E[:, rows[1][1], :] + J_E[:, rows[1][2], :]
    )
    J_E9[:, 8, :] = (
        J_E[:, rows[2][0], :] + J_E[:, rows[2][1], :] + J_E[:, rows[2][2], :]
    )

    J_x9[:, 6, :] = (
        J_x[:, rows[0][0], :] + J_x[:, rows[0][1], :] + J_x[:, rows[0][2], :]
    )
    J_x9[:, 7, :] = (
        J_x[:, rows[1][0], :] + J_x[:, rows[1][1], :] + J_x[:, rows[1][2], :]
    )
    J_x9[:, 8, :] = (
        J_x[:, rows[2][0], :] + J_x[:, rows[2][1], :] + J_x[:, rows[2][2], :]
    )

    tmp = torch.eye(9, 9, dtype=torch.float, device=minimal_sample.device)

    for i in range(b):
        try:
            J_Ex[i, :, :] = -torch.inverse(J_E9[i, :, :]).mm(J_x9[i, :, :])
            tmp = J_E9[i, :, :]
        except Exception as e:
            J_Ex[i, :, :] = -torch.inverse(tmp).mm(
                J_x9[i, :, :]
            )  # or -torch.linalg.pinv(J_E9[i, :, :]).mm(J_x9[i, :, :])

    return J_Ex


############ DDN with constraint ############


class EssentialMatrixNode(EqConstDeclarativeNode):
    """Declarative Essential matrix estimation node constraint"""

    def __init__(self, device, ift):
        super().__init__()
        self.estimator = ns.EssentialMatrixEstimatorNister(device=device, ift=ift)

    def objective(self, minimal_samples, E_true, y):
        """
        minimal_samples : b x 5 x 4
        E_true: 3 x 3
        y : b x 3 x 3
        """
        batch_size = minimal_samples[:, :, :2].shape[0]

        w = torch.ones(
            (batch_size, 5), dtype=torch.float, device=minimal_samples.device
        ) / float(5.0)

        A_ = (
            torch.cat(
                (
                    minimal_samples[:, :, :2],
                    torch.ones((batch_size, 5, 1), device=minimal_samples.device),
                ),
                -1,
            )
        ).unsqueeze(-2)
        B_ = (
            torch.cat(
                (
                    minimal_samples[:, :, 2:],
                    torch.ones((batch_size, 5, 1), device=minimal_samples.device),
                ),
                -1,
            )
        ).unsqueeze(-1)
        M = A_ * B_  # [8 x 5 x 3 x 3]
        res = ((M.view(batch_size, -1, 9)).matmul(y.view(batch_size, 9, 1))) ** 2
        out = torch.einsum("bn,bn->b", (w, res.squeeze(-1)))
        return out

    def equality_constraints(self, minimal_samples, E_true, y):
        E_Et = y.matmul(y.permute(0, 2, 1))
        E_Et_trace = torch.einsum("bii->b", E_Et)
        eq_constr1 = 2 * E_Et.matmul(y) - torch.einsum("b,bnm->bnm", E_Et_trace, y)
        eq_constr1 = (eq_constr1**2).sum(dim=(-1, -2))

        eq_constr2 = (y.view(-1, 9) ** 2).sum(dim=-1) - 1.0

        return torch.cat((eq_constr1.unsqueeze(1), eq_constr2.unsqueeze(1)), 1)

    def solve(self, minimal_samples, E_true):
        minimal_samples = minimal_samples.detach()
        y = self._solve_(minimal_samples, E_true).requires_grad_()
        return y.detach(), None

    def _solve_(self, minimal_samples, E_true):
        """
        minimal_samples : b x 5 x 4
        E_true: 3 x 3
        """
        est = self.estimator.estimate_minimal_model(minimal_samples)

        solution_num = 10

        distances = torch.norm(
            est - E_true.unsqueeze(0).repeat(est.shape[0], 1, 1), dim=(1, 2)
        ).view(est.shape[0], -1)

        try:
            chosen_indices = torch.argmin(distances.view(-1, solution_num), dim=-1)
            chosen_models = torch.stack(
                [
                    (est.view(-1, solution_num, 3, 3))[i, chosen_indices[i], :]
                    for i in range(int(est.shape[0] / solution_num))
                ]
            )

        except ValueError as e:
            print(
                "not enough models for selection, we choose the first solution in this batch",
                e,
                est.shape,
            )
            chosen_models = est[0].unsqueeze(0)

        return chosen_models
