import torch
import numpy as np

def my_atan2(y, x):
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y / (x + 1e-6))
    ans += ((y > 0) & (x < 0)) * pi
    ans -= ((y < 0) & (x < 0)) * pi
    ans *= (1 - ((y > 0) & (x == 0)) * 1.0)
    ans += ((y > 0) & (x == 0)) * (pi / 2)
    ans *= (1 - ((y < 0) & (x == 0)) * 1.0)
    ans += ((y < 0) & (x == 0)) * (-pi / 2)
    return ans