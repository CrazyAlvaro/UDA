from typing import Optional
import torch
import torch.nn as nn

# https://github.com/thuml/Transfer-Learning-Library/blob/7b0ccb3a8087ecc65daf4b1e815e5a3f42106641/dalib/modules/kernels.py#L13
# https://tl.thuml.ai/dalib/modules.html

class GaussianKernel(nn.Module):
    """
    Gaussian Kernel Matrix
    k(x1, x2) = exp( - ||x1 - x2||**2 / (2*sigma**2) )

    where x1, x2 R**d are 1-d tensors
    """
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True, 
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        # assert track_running_stats or sigma is not None
        assert sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        # self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        # if self.track_running_stats:
        #    self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))
