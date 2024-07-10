import torch


def create_toy_dataset():
    p1 = torch.Tensor([1, 0, 0]).unsqueeze(0)
    p2 = torch.Tensor([0.0472, 0.9299, 0.7934]).unsqueeze(0)
    p3 = torch.Tensor([0.7017, 0.1494, 0.7984]).unsqueeze(0)
    p4 = torch.Tensor([0.6007, 0.8878, 0.9169]).unsqueeze(0)
    P = torch.cat([p1, p2, p3, p4], dim=0).permute(1, 0).unsqueeze(0)  # 1 x 3 x 4

    q1 = torch.Tensor([0.9, 0.1, 0.0]).unsqueeze(0)
    q2 = torch.Tensor([0.0472, 0.9299, 0.7934]).unsqueeze(0)
    q3 = torch.Tensor([0.7017, 0.1494, 0.7984]).unsqueeze(0)
    q4 = torch.Tensor([0.6007, 0.8878, 0.9169]).unsqueeze(0)
    Q = torch.cat([q1, q2, q3, q4], dim=0).permute(1, 0).unsqueeze(0)  # 1 x 3 x 4

    R_true = torch.eye(3, dtype=torch.float).unsqueeze(0)  # 1 x 3 x 3

    return P, Q, R_true


def get_dataset():
    return create_toy_dataset()
