import torch

# https://github.com/thuml/Transfer-Learning-Library/blob/7b0ccb3a8087ecc65daf4b1e815e5a3f42106641/common/utils/metric/__init__.py

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the accuracy for binary classification
    """
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct
