# methods/tv.py

import torch


def total_variation_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic RGB Total Variation.

    x shape:
        [B, 3, H, W]
    """
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()

    return tv_h + tv_w


def tv_reconstruction(
    degraded: torch.Tensor,
    K,
    lambda_tv: float = 0.01,
    num_iters: int = 300,
    lr: float = 0.05,
    clamp: bool = True,
    verbose: bool = True,
) -> torch.Tensor:
    """
    RGB TV reconstruction.

    Minimizes:

        ||K(x) - y||^2 + lambda_tv * TV(x)

    where:
        y = degraded observation
        K = RGB blur operator
        x = reconstructed clean image
    """
    x = degraded.clone().detach()
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    for it in range(num_iters):
        optimizer.zero_grad()

        data_term = torch.mean((K(x) - degraded) ** 2)
        tv_term = total_variation_rgb(x)

        loss = data_term + lambda_tv * tv_term

        loss.backward()
        optimizer.step()

        if clamp:
            with torch.no_grad():
                x.clamp_(0.0, 1.0)

        if verbose and (it % 50 == 0 or it == num_iters - 1):
            print(
                f"[TV] iter {it:04d} | "
                f"loss={loss.item():.6f} | "
                f"data={data_term.item():.6f} | "
                f"tv={tv_term.item():.6f}"
            )

    return x.detach()