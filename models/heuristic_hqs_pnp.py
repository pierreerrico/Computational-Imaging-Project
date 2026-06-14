import os
import numpy as np
import torch
import torch.nn as nn

from IPPy.utilities.metrics import PSNR, SSIM, RE
from utilities.plotter import plot 


class DRUNetDenoiser(nn.Module):
    """
    Wrapper for DRUNet denoising.

    DRUNet receives a 4-channel tensor made of the RGB image and
    a constant sigma map.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(
                sigma,
                device=x.device,
                dtype=x.dtype,
            )

        sigma = sigma.to(device=x.device, dtype=x.dtype)

        sigma_map = torch.ones(
            x.shape[0],
            1,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        ) * sigma

        x_sigma = torch.cat([x, sigma_map], dim=1)

        return self.model(x_sigma)


class HQSCG:
    """
    Conjugate-gradient solver for the HQS data step.

    It solves

        (K^T K + rho I)x = K^T y_d + rho z

    where z is the current denoised estimate.
    """

    def __init__(self, K) -> None:
        self.K = K

    def batch_dot(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(a * b, dim=(1, 2, 3), keepdim=True)

    def batch_norm(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sqrt(torch.sum(x * x, dim=(1, 2, 3), keepdim=True))

    def apply_normal_operator(
        self,
        x: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        return self.K.T(self.K(x)) + rho * x

    def __call__(
        self,
        y_d: torch.Tensor,
        z: torch.Tensor,
        rho: float | torch.Tensor,
        starting_point: torch.Tensor | None = None,
        maxiter: int = 30,
        tol: float = 1e-5,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        y_d = y_d.float()
        z = z.float()

        if not torch.is_tensor(rho):
            rho = torch.tensor(
                rho,
                device=y_d.device,
                dtype=torch.float32,
            )

        rho = rho.float().view(1, 1, 1, 1)

        if starting_point is None:
            x = z.clone()
        else:
            x = starting_point.float().clone()

        b = self.K.T(y_d) + rho * z

        r = b - self.apply_normal_operator(x, rho)
        p = r.clone()

        rs_old = self.batch_dot(r, r)
        rs_initial = rs_old.clone()

        info: dict[str, list[float]] = {
            "relative_system_residual": [],
            "data_residual": [],
        }

        eps = 1e-8

        for k in range(maxiter):
            relative_system_residual = torch.sqrt(
                rs_old / (rs_initial + eps)
            ).mean()

            data_residual = self.batch_norm(
                self.K(x) - y_d
            ).mean()

            info["relative_system_residual"].append(
                relative_system_residual.item()
            )
            info["data_residual"].append(data_residual.item())

            if verbose:
                print(
                    f"CG iter {k:03d} | "
                    f"rel_sys_res={relative_system_residual.item():.6f} | "
                    f"data_res={data_residual.item():.4f}"
                )

            if relative_system_residual.item() < tol:
                break

            Ap = self.apply_normal_operator(p, rho)
            alpha = rs_old / (self.batch_dot(p, Ap) + eps)

            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = self.batch_dot(r, r)
            beta = rs_new / (rs_old + eps)

            p = r + beta * p
            rs_old = rs_new

        return x, info


class PnPHQSRegularizer:
    """
    Plug-and-Play HQS reconstruction using a frozen denoiser.

    Parameters
    ----------
    denoiser : nn.Module
        Denoising model used as prior. It must accept denoiser(x, sigma).

    schedules : dict[str, dict] | None
        HQS schedules to test.

    cg_tol : float
        Stopping tolerance for conjugate gradient.

    init_mode : str
        Initial estimate. Supported values are "degraded" and "adjoint".
    """

    def __init__(
        self,
        denoiser: nn.Module,
        schedules: dict[str, dict] | None = None,
        cg_tol: float = 1e-5,
        init_mode: str = "degraded",
    ) -> None:
        self.denoiser = denoiser
        self.schedules = schedules or {
            "soft_25_15": {
                "iter_num": 5,
                "model_sigma_1": 25.0,
                "model_sigma_2": 15.0,
                "w": 1.0,
                "rho_scale": 0.23,
                "cg_iters": 30,
            },
            "medium_35_15": {
                "iter_num": 6,
                "model_sigma_1": 35.0,
                "model_sigma_2": 15.0,
                "w": 1.0,
                "rho_scale": 0.23,
                "cg_iters": 30,
            },
            "strong_49_final": {
                "iter_num": 8,
                "model_sigma_1": 49.0,
                "model_sigma_2": None,
                "w": 1.0,
                "rho_scale": 0.23,
                "cg_iters": 30,
            },
        }
        self.cg_tol = cg_tol
        self.init_mode = init_mode

        self.denoiser.eval()
        for p in self.denoiser.parameters():
            p.requires_grad = False

    def get_rhos_and_sigmas(
        self,
        noise_level: float,
        schedule: dict,
    ) -> tuple[list[float], list[float]]:
        model_sigma_2 = schedule["model_sigma_2"]

        if model_sigma_2 is None:
            model_sigma_2 = noise_level * 255.0

        sigma_log = np.logspace(
            np.log10(schedule["model_sigma_1"]),
            np.log10(model_sigma_2),
            schedule["iter_num"],
        ).astype(np.float32)

        sigma_lin = np.linspace(
            schedule["model_sigma_1"],
            model_sigma_2,
            schedule["iter_num"],
        ).astype(np.float32)

        sigmas = (
            sigma_log * schedule["w"]
            + sigma_lin * (1.0 - schedule["w"])
        ) / 255.0

        rhos = [
            schedule["rho_scale"] * (noise_level ** 2) / (sigma ** 2)
            for sigma in sigmas
        ]

        return rhos, sigmas.tolist()

    def initialise(
        self,
        y_d: torch.Tensor,
        K,
    ) -> torch.Tensor:
        if self.init_mode == "degraded":
            return y_d.clone().clamp(0.0, 1.0)

        if self.init_mode == "adjoint":
            return K.T(y_d).clamp(0.0, 1.0)

        raise ValueError("init_mode must be either 'degraded' or 'adjoint'.")

    def reconstruct_single_schedule(
        self,
        y_d: torch.Tensor,
        K,
        noise_level: float,
        schedule: dict,
    ) -> tuple[torch.Tensor, dict[str, list]]:
        rhos, sigmas = self.get_rhos_and_sigmas(
            noise_level=noise_level,
            schedule=schedule,
        )

        cg_solver = HQSCG(K)
        x = self.initialise(y_d, K)

        info = {
            "x": [x.detach().cpu()],
            "x_data": [],
            "z": [],
            "rho": [],
            "sigma": [],
            "cg_infos": [],
        }

        for rho_value, sigma_value in zip(rhos, sigmas):
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=False):  # type: ignore
                    x_data, cg_info = cg_solver(
                        y_d=y_d.float(),
                        z=x.float(),
                        rho=rho_value,
                        starting_point=x.float(),
                        maxiter=schedule["cg_iters"],
                        tol=self.cg_tol,
                        verbose=False,
                    )

                x_data = x_data.clamp(0.0, 1.0)
                x = self.denoiser(x_data, sigma=sigma_value)
                x = x.clamp(0.0, 1.0)

            info["x_data"].append(x_data.detach().cpu())
            info["z"].append(x.detach().cpu())
            info["x"].append(x.detach().cpu())
            info["rho"].append(torch.tensor(rho_value).detach().cpu())
            info["sigma"].append(torch.tensor(sigma_value).detach().cpu())
            info["cg_infos"].append(cg_info)

        return x, info

    def __call__(
        self,
        y_d: torch.Tensor,
        K,
        noise_level: float,
        x_gt: torch.Tensor | None = None,
        save_dir: str | None = None,
        preview: bool = False,
    ) -> list[dict]:
        """
        Reconstruct an image using Plug-and-Play HQS.

        Parameters
        ----------
        y_d : torch.Tensor
            Degraded image with shape [B, C, H, W].

        K : operator
            Forward operator used to generate the measurements.

        noise_level : float
            Noise level used in the degradation.

        x_gt : torch.Tensor | None, optional
            Ground-truth image used only for computing PSNR, SSIM and RE.

        save_dir : str | None, optional
            Directory where reconstruction files are saved.

        preview : bool
            If True, plot degraded image, reconstruction and ground truth.

        Returns
        -------
        list[dict]
            One result dictionary for each schedule.
        """
        if y_d.ndim != 4:
            raise ValueError(f"Expected y_d with shape [B,C,H,W], got {y_d.shape}")

        if x_gt is not None and x_gt.shape != y_d.shape:
            raise ValueError(
                f"x_gt and y_d must have the same shape, got {x_gt.shape} and {y_d.shape}"
            )

        device = y_d.device
        y_d = y_d.detach().to(device).float()

        if x_gt is not None:
            x_gt = x_gt.detach().to(device).float()

        self.denoiser.to(device)
        self.denoiser.eval()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        results = []

        for schedule_name, schedule in self.schedules.items():
            print(f"Running PnP-HQS reconstruction with schedule = {schedule_name}")

            x_hat, info = self.reconstruct_single_schedule(
                y_d=y_d,
                K=K,
                noise_level=noise_level,
                schedule=schedule,
            )

            x_hat = x_hat.clamp(0.0, 1.0).detach()

            result = {
                "schedule": schedule_name,
                "reconstruction": x_hat,
                "info": info,
            }

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
                        "schedule": schedule_name,
                        "reconstruction": x_hat.detach().cpu(),
                        "psnr": result.get("psnr"),
                        "ssim": result.get("ssim"),
                        "re": result.get("re"),
                    },
                    os.path.join(save_dir, f"pnp_hqs_{schedule_name}.pt"),
                )

            if preview:
                if x_gt is not None:
                    plot(
                        y_d[0].detach().cpu(),
                        x_hat[0].detach().cpu(),
                        x_gt[0].detach().cpu(),
                        titles=[
                            "Degraded image",
                            f"PnP-HQS reconstruction {schedule_name}",
                            "Ground truth",
                        ],
                    )
                else:
                    plot(
                        y_d[0].detach().cpu(),
                        x_hat[0].detach().cpu(),
                        titles=[
                            "Degraded image",
                            f"PnP-HQS reconstruction {schedule_name}",
                        ],
                    )

            results.append(result)

        return results
