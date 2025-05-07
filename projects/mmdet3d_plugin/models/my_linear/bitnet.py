from torch import nn, Tensor
import torch.nn.functional as F
# from torch_ext_test import my_func_mm_bc_add

class SimpleRMSNorm(nn.Module):
    """
    SimpleRMSNorm

    Args:
        dim (int): dimension of the embedding

    Usage:
    We can use SimpleRMSNorm as a layer in a neural network as follows:
        >>> x = torch.randn(1, 10, 512)
        >>> simple_rms_norm = SimpleRMSNorm(dim=512)
        >>> simple_rms_norm(x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm"""
        return F.normalize(x, dim=-1) * self.scale

def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor):
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class BitLinear(nn.Module):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """

    def forward(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
       
        x_norm = SimpleRMSNorm(x.shape[-1])(x)

        # STE using detach
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        b_quant = b + (weight_quant(b) - b).detach()
        y = F.linear(x_quant, w_quant, b_quant)
        return y
    


