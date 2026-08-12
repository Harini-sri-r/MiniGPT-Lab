import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """Add a sublayer's output to its original input."""

    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x):
        # The sublayer transforms x, such as attention or a feed-forward
        # network. The original x is kept as a direct information path.
        sublayer_output = self.sublayer(x)

        if not isinstance(sublayer_output, torch.Tensor):
            raise TypeError("sublayer must return a torch.Tensor.")

        # Addition is only meaningful when corresponding elements align.
        if x.shape != sublayer_output.shape:
            raise ValueError(
                "Residual connection requires matching shapes: "
                f"input has {tuple(x.shape)}, but sublayer output has "
                f"{tuple(sublayer_output.shape)}."
            )

        return x + sublayer_output
