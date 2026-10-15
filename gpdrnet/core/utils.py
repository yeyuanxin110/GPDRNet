# Inference-only adaptation of RoMa (Johan Edstedt, MIT).
# Modified for standalone GPDRNet; see README.md (Attribution and release license).
import math
import torch


@torch.no_grad()
def cls_to_flow_refine(cls):
    (B, C, H, W) = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1 + 1 / res, 1 - 1 / res, steps=res, device=device) for _ in range(2)],
        indexing="ij",
    )
    G = torch.stack([G[1], G[0]], dim=-1).reshape(C, 2)
    if device.type == "mps":
        cls = cls.log_softmax(dim=1).exp()
    else:
        cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices
    index = (
        torch.stack((mode - 1, mode, mode + 1, mode - res, mode + res), dim=1)
        .clamp(0, C - 1)
        .long()
    )
    neighbours = torch.gather(cls, dim=1, index=index)[..., None]
    flow = (
        neighbours[:, 0] * G[index[:, 0]]
        + neighbours[:, 1] * G[index[:, 1]]
        + neighbours[:, 2] * G[index[:, 2]]
        + neighbours[:, 3] * G[index[:, 3]]
        + neighbours[:, 4] * G[index[:, 4]]
    )
    tot_prob = neighbours.sum(dim=1)
    flow = flow / tot_prob
    return flow


def get_grid(b, h, w, device):
    grid = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (b, h, w)], indexing="ij"
    )
    grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
    return grid


def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        autocast_device = str(device).split(":")[0]
    if "cuda" in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        autocast_device = "cpu"
    return (autocast_device, enabled, out_dtype)
