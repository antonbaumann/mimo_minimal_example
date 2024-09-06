import torch
import torch.nn as nn


class MCSM(nn.Module):
    """
    Compute the softmax of a Gaussian distribution using Monte Carlo sampling.

    Args:
        mean: [*batch, C, H, W]
        std: [*batch, C, H, W]
        num_samples: Number of samples to use for Monte Carlo sampling.
        log: Whether to return the log softmax.
    Returns:
        softmax: [*batch, C, H, W]
    """

    def __init__(self,num_samples,log):
        super(MCSM, self).__init__()
        self.num_samples = num_samples
        self.log = log

    def forward(self,
                logits: torch.Tensor,
                std: torch.Tensor):
        
        softmax_sum = torch.zeros_like(logits)
        for _ in range(self.num_samples):
            # [*batch, C, H, W]
            epsilon = torch.randn(std.shape, device=std.device)

            # [*batch, C, H, W]
            logit_samples = logits + epsilon * std
            softmax_sum += torch.nn.functional.softmax(logit_samples, dim=-3)

        # [*batch, C, H, W]
        softmax = softmax_sum / self.num_samples

        if self.log:
            return torch.log(softmax)

        return softmax


class MCCE(nn.Module):
    def __init__(
            self, 
            num_samples: int,
            label_smoothing: float = 0.0,
        ):
        super(MCCE, self).__init__()
        self.num_samples = num_samples
        self.label_smoothing = label_smoothing

    def forward(
            self,
            logits: torch.Tensor,
            std: torch.Tensor,
            labels: torch.Tensor,
            weight: torch.Tensor = None,
        ):
        
        loss = 0
        for _ in range(self.num_samples):
            # [*batch, C, H, W]
            epsilon = torch.randn(std.shape, device=std.device)

            # [*batch, C, H, W]
            logit_samples = logits + epsilon * std

            # compute cross-entropy loss for each pixel
            # [*batch, H, W]
            loss_per_pixel = torch.nn.functional.cross_entropy(
                input=logit_samples,
                target=labels,
                reduction='none',
                label_smoothing=self.label_smoothing,
            )

            # apply per-pixel weights
            if weight is not None:
                loss_per_pixel *= weight

            # average loss over spatial dimensions
            loss += loss_per_pixel.mean() / self.num_samples

        return loss