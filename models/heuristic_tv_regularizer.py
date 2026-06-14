import os
import torch
from IPPy.utilities.metrics import PSNR, SSIM, RE
from IPPy import operators as op, solvers as sol
from utilities.plotter import plot

class TotalVariationRegularizer:
    """
    Total Variation reconstruction using the
    Chambolle-Pock unconstrained solver from IPPy.

    Parameters
    ----------
    lambda_values : list[float]
        Values of λ to test.

    max_iters : int
        Maximum number of iterations for each reconstruction.
    """

    def __init__(
        self,
        lambda_values: list[float] | None = None,
        max_iters: int = 100,
    ) -> None:
        self.lambda_values = lambda_values or [
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            1e-1,
            3e-1,
        ]
        self.max_iters = max_iters

    def __call__(
        self,
        y_d: torch.Tensor,
        K: op.Operator,
        x_gt: torch.Tensor | None = None,
        save_dir: str | None = None,
        preview: bool = False,
    ) -> list[dict]:
        """
        Reconstruct an image using TV regularization.

        Parameters
        ----------
        y_d : torch.Tensor
            Degraded image with shape [B, C, H, W].

        K : op.Operator
            Forward operator used to generate the measurements.

        x_gt : torch.Tensor | None, optional
            Ground-truth image used only for computing
            PSNR, SSIM and RE. If None, no metrics are computed.

        save_dir : str | None, optional
            Directory where reconstruction files are saved.

        preview : bool
            If True, plot degraded image, reconstruction and ground truth.

        Returns
        -------
        list[dict]
            One result dictionary for each λ value.
        """

        # IPPy operators are safer on CPU.
        y_d = y_d.detach().cpu()

        if x_gt is not None:
            x_gt = x_gt.detach().cpu()

        if y_d.ndim != 4:
            raise ValueError(f"Expected y_d with shape [B,C,H,W], got {y_d.shape}")

        if x_gt is not None and x_gt.shape != y_d.shape:
            raise ValueError(
                f"x_gt and y_d must have the same shape, got {x_gt.shape} and {y_d.shape}"
            )

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        solver = sol.ChambollePockTpVUnconstrained(K)

        results = []

        for lambda_value in self.lambda_values:
            print(f"Running TV reconstruction with lambda = {lambda_value:.1e}")

            restored_channels = []
            infos = []

            for c in range(y_d.shape[1]):
                y_d_c = y_d[:, c:c + 1, :, :].detach()

                x_gt_c = (
                    x_gt[:, c:c + 1, :, :].detach()
                    if x_gt is not None
                    else None
                )

                x_hat_c, info = solver(
                    y_d_c,
                    x_true=x_gt_c,
                    starting_point=torch.zeros_like(y_d_c),
                    lmbda=lambda_value,
                    maxiter=self.max_iters,
                    p=1,
                    verbose=False,
                )

                restored_channels.append(x_hat_c.detach())
                infos.append(info)

            x_hat = torch.cat(restored_channels, dim=1)
            x_hat = x_hat.clamp(0.0, 1.0).detach()

            result = {
                "lambda": lambda_value,
                "reconstruction": x_hat,
                "infos": infos,
            }

            lambda_name = f"{lambda_value:.0e}".replace("-", "m")

            if x_gt is not None:
                result["psnr"] = PSNR(x_hat, x_gt)
                result["ssim"] = SSIM(x_hat, x_gt)
                result["re"] = RE(x_hat, x_gt)

                print(
                    f"Done | "
                    f"PSNR={result['psnr']:.2f} dB | "
                    f"SSIM={result['ssim']:.4f} | "
                    f"RE={result['re']:.4f}"
                )
            else:
                print("Done", flush=True)

            if save_dir is not None:
                torch.save(
                    {
                        "lambda": lambda_value,
                        "reconstruction": x_hat.detach().cpu(),
                        "psnr": result.get("psnr"),
                        "ssim": result.get("ssim"),
                        "re": result.get("re"),
                    },
                    os.path.join(save_dir, f"tv_lambda_{lambda_name}.pt"),
                )

            if preview:
                if x_gt is not None:
                    plot(
                        y_d[0].detach().cpu(),
                        x_hat[0].detach().cpu(),
                        x_gt[0].detach().cpu(),
                        titles=[
                            "Degraded image",
                            f"TV reconstruction λ={lambda_value:.1e}",
                            "Ground truth",
                        ],
                    )
                else:
                    plot(
                        y_d[0].detach().cpu(),
                        x_hat[0].detach().cpu(),
                        titles=[
                            "Degraded image",
                            f"TV reconstruction λ={lambda_value:.1e}",
                        ],
                    )

            results.append(result)

        return results