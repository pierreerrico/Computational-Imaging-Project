import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from IPPy.utilities.metrics import PSNR, SSIM
from utilities.plotter import plot
from utilities.degradation import DegradationParameters, ImageDegradation, RGBBlurOperator


class DRUNetPrior(nn.Module):
    """
    Wrapper for a DRUNet-like denoiser that expects a 4-channel input.

    The input image has shape [B, 3, H, W]. The wrapper adds a fourth channel
    containing the denoising level sigma, producing [B, 4, H, W].
    """

    def __init__(
        self,
        model: nn.Module,
        freeze: bool = True,
        use_mixed_precision: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.use_mixed_precision = use_mixed_precision

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        nn.Module.train(self, False)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "DRUNetPrior":
        nn.Module.train(self, False)
        self.model.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma_map = self.make_sigma_map(x, sigma)
        x_in = torch.cat([x, sigma_map], dim=1)

        use_amp = self.use_mixed_precision and x_in.is_cuda

        with torch.amp.autocast( # type: ignore
            "cuda",
            enabled=use_amp,
        ):
            out = self.model(x_in)

        return out.float()

    @staticmethod
    def make_sigma_map(
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
            sigma = sigma.to(device=x.device, dtype=x.dtype)

            if sigma.dim() == 0 or sigma.numel() == 1:
                return sigma.reshape(1, 1, 1, 1).expand(
                    x.shape[0],
                    1,
                    x.shape[2],
                    x.shape[3],
                )

            if sigma.shape == (x.shape[0],):
                return sigma.reshape(x.shape[0], 1, 1, 1).expand(
                    -1,
                    1,
                    x.shape[2],
                    x.shape[3],
                )

            if sigma.shape == (x.shape[0], 1, 1, 1):
                return sigma.expand(
                    -1,
                    1,
                    x.shape[2],
                    x.shape[3],
                )

            if sigma.shape == (x.shape[0], 1, x.shape[2], x.shape[3]):
                return sigma

            raise ValueError(
                f"Unsupported sigma shape {sigma.shape} for input shape {x.shape}."
            )


class DataConsistencyStep(nn.Module):
    """
    Single learned gradient step for the data-consistency term.

    This replaces the conjugate-gradient solve with one explicit step:

        x = z - alpha * K^T(Kz - y)

    It is cheaper than CG because each layer applies K and K^T only once.
    """

    def __init__(
        self,
        K: RGBBlurOperator,
    ) -> None:
        super().__init__()
        self.K = K

    def forward(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        y = y.float()
        z = z.float()
        alpha = alpha.to(device=y.device, dtype=y.dtype).reshape(1, 1, 1, 1)

        residual = self.K(z) - y
        gradient = self.K.T(residual)

        return z - alpha * gradient


class HQSNet(nn.Module):
    """
    Unrolled HQS-inspired restoration model with a frozen DRUNet prior.

    Each layer performs:
        1. one learned data-consistency gradient step;
        2. one frozen DRUNet prior step.

    The trainable parameters are:
        - alpha_k: data-consistency step size per layer;
        - sigma_k: DRUNet denoising level per layer.
    """

    def __init__(
        self,
        K: RGBBlurOperator,
        prior: DRUNetPrior,
        n_layers: int = 6,
        initial_alpha: float = 0.1,
        initial_sigma: float = 25.0 / 255.0,
        min_alpha: float = 1e-6,
        min_sigma: float = 1e-6,
        init_mode: str = "degraded",
    ) -> None:
        super().__init__()

        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")

        if initial_alpha <= min_alpha:
            raise ValueError("initial_alpha must be greater than min_alpha.")

        if initial_sigma <= min_sigma:
            raise ValueError("initial_sigma must be greater than min_sigma.")

        if init_mode not in {"degraded", "adjoint"}:
            raise ValueError("init_mode must be either 'degraded' or 'adjoint'.")

        self.K = K
        self.prior = prior
        self.n_layers = n_layers
        self.min_alpha = min_alpha
        self.min_sigma = min_sigma
        self.init_mode = init_mode

        self.data_step = DataConsistencyStep(
            K=K,
        )

        self.raw_alpha = nn.Parameter(
            torch.full(
                size=(n_layers,),
                fill_value=self._inverse_softplus(initial_alpha - min_alpha),
                dtype=torch.float32,
            )
        )

        self.raw_sigma = nn.Parameter(
            torch.full(
                size=(n_layers,),
                fill_value=self._inverse_softplus(initial_sigma - min_sigma),
                dtype=torch.float32,
            )
        )

        self.freeze_prior()

    @staticmethod
    def _inverse_softplus(x: float) -> float:
        x = max(float(x), 1e-12)
        return math.log(math.expm1(x))

    def freeze_prior(self) -> None:
        self.prior.freeze()

    def train(self, mode: bool = True) -> "HQSNet":
        super().train(mode)
        self.freeze_prior()
        return self

    def get_alphas(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha) + self.min_alpha

    def get_sigmas(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + self.min_sigma

    def initialize(self, y: torch.Tensor) -> torch.Tensor:
        if self.init_mode == "degraded":
            return y.clone()

        return self.K.T(y)

    def forward(
        self,
        y: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        y = y.float()

        z = self.initialize(y)

        alphas = self.get_alphas()
        sigmas = self.get_sigmas()

        intermediates: dict[str, list[torch.Tensor]] = {
            "x_data": [],
            "z_prior": [],
            "alpha": [],
            "sigma": [],
        }

        for layer_index in range(self.n_layers):
            alpha = alphas[layer_index]
            sigma = sigmas[layer_index]

            x = self.data_step(
                y=y,
                z=z,
                alpha=alpha,
            )

            z = self.prior(
                x=x,
                sigma=sigma,
            )

            if return_intermediates:
                intermediates["x_data"].append(x.detach())
                intermediates["z_prior"].append(z.detach())
                intermediates["alpha"].append(alpha.detach())
                intermediates["sigma"].append(sigma.detach())

        if return_intermediates:
            return z, intermediates

        return z

    def clipped_forward(self, y: torch.Tensor) -> torch.Tensor:
        pred = self.forward(y, return_intermediates=False)

        if isinstance(pred, tuple):
            pred = pred[0]

        return pred.clamp(0.0, 1.0)

    def parameter_summary(self) -> dict[str, list[float]]:
        sigmas = self.get_sigmas().detach().cpu().tolist()

        return {
            "alpha": self.get_alphas().detach().cpu().tolist(),
            "sigma": sigmas,
            "sigma_255": [sigma * 255.0 for sigma in sigmas],
        }


class HQSNetTrainer:
    """
    Training wrapper for an unrolled HQS restoration model.

    The model is trained on supervised image restoration:

        clean image -> degradation -> degraded image -> HQSNet -> restored image
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def train_model(
        self,
        n_epochs: int = 50,
        train_dataset=None,
        validation_dataset=None,
        train_degradation: ImageDegradation | None = None,
        validation_degradations: list[ImageDegradation] | None = None,
        max_validation_batches: int | None = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        checkpoint_path: str = "./weights/HQSNet/HQS_checkpoint.pth",
        resume: bool = True,
        preview_every: int | None = 5,
        preview_n: int | None = 2,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> dict[str, list[float]]:

        device = torch.device(device)
        self.model.to(device)

        if train_dataset is None:
            raise ValueError("Training dataset must be defined.")

        if validation_dataset is None:
            raise ValueError("Validation dataset must be defined.")

        if train_degradation is None:
            train_degradation = ImageDegradation(
                DegradationParameters(
                    image_size=256,
                    kernel_type="motion",
                    kernel_size=9,
                    motion_angle=45.0,
                    noise_levels=[0.005, 0.01, 0.05, 0.1],
                )
            )

        if validation_degradations is None:
            validation_degradations = [
                ImageDegradation(
                    DegradationParameters(
                        image_size=256,
                        kernel_type="motion",
                        kernel_size=9,
                        motion_angle=45.0,
                        noise_levels=[0.005],
                    )
                ),
                ImageDegradation(
                    DegradationParameters(
                        image_size=256,
                        kernel_type="motion",
                        kernel_size=9,
                        motion_angle=45.0,
                        noise_levels=[0.01],
                    )
                ),
                ImageDegradation(
                    DegradationParameters(
                        image_size=256,
                        kernel_type="motion",
                        kernel_size=9,
                        motion_angle=45.0,
                        noise_levels=[0.05],
                    )
                ),
                ImageDegradation(
                    DegradationParameters(
                        image_size=256,
                        kernel_type="motion",
                        kernel_size=9,
                        motion_angle=45.0,
                        noise_levels=[0.1],
                    )
                ),
            ]

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if len(trainable_parameters) == 0:
            raise ValueError("The model has no trainable parameters.")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=0.0,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

        loss_l1 = nn.L1Loss()

        history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_psnr": [],
            "validation_ssim": [],
            "learning_rate": [],
        }

        best_validation_loss = float("inf")
        start_epoch = 0

        if resume and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])

            history = checkpoint["history"]
            history.setdefault("learning_rate", [])

            best_validation_loss = checkpoint["best_validation_loss"]
            start_epoch = checkpoint["epoch"] + 1

            print(f"Resumed HQSNet training from epoch {start_epoch}")

        for epoch in range(start_epoch, n_epochs):
            self.model.train()

            train_losses = []

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{n_epochs}",
                leave=True,
            )

            for clean in progress_bar:
                clean = clean.to(device, non_blocking=True).float()
                degraded = train_degradation(clean).float()

                optimizer.zero_grad(set_to_none=True)

                pred = self.model(degraded)

                if isinstance(pred, tuple):
                    pred = pred[0]

                loss = loss_l1(pred, clean)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=1.0,
                )

                optimizer.step()

                train_losses.append(loss.item())

                avg_loss = sum(train_losses) / len(train_losses)
                current_lr = optimizer.param_groups[0]["lr"]

                progress_bar.set_postfix(
                    batch_loss=f"{loss.item():.6f}",
                    avg_loss=f"{avg_loss:.6f}",
                    lr=f"{current_lr:.2e}",
                )

            mean_train_loss = sum(train_losses) / len(train_losses)
            history["train_loss"].append(mean_train_loss)

            self.model.eval()

            validation_losses = []
            validation_psnr_values = []
            validation_ssim_values = []

            with torch.no_grad():
                for validation_degradation in validation_degradations:
                    for batch_idx, clean in enumerate(validation_loader):
                        if (
                            max_validation_batches is not None
                            and batch_idx >= max_validation_batches
                        ):
                            break

                        clean = clean.to(device, non_blocking=True).float()
                        degraded = validation_degradation(clean).float()

                        pred = self.model(degraded)

                        if isinstance(pred, tuple):
                            pred = pred[0]

                        val_loss = loss_l1(pred, clean)
                        pred_clamped = pred.clamp(0.0, 1.0)

                        psnr_value = PSNR(pred_clamped.float(), clean.float())
                        ssim_value = SSIM(pred_clamped.float(), clean.float())

                        if isinstance(psnr_value, torch.Tensor):
                            psnr_value = psnr_value.item()

                        if isinstance(ssim_value, torch.Tensor):
                            ssim_value = ssim_value.item()

                        validation_losses.append(val_loss.item())
                        validation_psnr_values.append(psnr_value)
                        validation_ssim_values.append(ssim_value)

            mean_validation_loss = sum(validation_losses) / len(validation_losses)
            mean_validation_psnr = sum(validation_psnr_values) / len(validation_psnr_values)
            mean_validation_ssim = sum(validation_ssim_values) / len(validation_ssim_values)

            scheduler.step(mean_validation_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            history["validation_loss"].append(mean_validation_loss)
            history["validation_psnr"].append(mean_validation_psnr)
            history["validation_ssim"].append(mean_validation_ssim)
            history["learning_rate"].append(current_lr)

            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"train_loss: {mean_train_loss:.6f} | "
                f"val_loss: {mean_validation_loss:.6f} | "
                f"val_PSNR: {mean_validation_psnr:.4f} | "
                f"val_SSIM: {mean_validation_ssim:.4f} | "
                f"lr: {current_lr:.2e}"
            )

            if hasattr(self.model, "parameter_summary"):
                print(self.model.parameter_summary())  # type: ignore

            if (
                preview_every is not None
                and preview_n is not None
                and (epoch + 1) % preview_every == 0
            ):
                preview_clean = next(iter(validation_loader))
                preview_clean = preview_clean[:preview_n].to(
                    device,
                    non_blocking=True,
                ).float()

                with torch.no_grad():
                    preview_degraded = validation_degradations[0](
                        preview_clean
                    ).float()

                    preview_pred = self.model(preview_degraded)

                    if isinstance(preview_pred, tuple):
                        preview_pred = preview_pred[0]

                    preview_pred = preview_pred.clamp(0.0, 1.0)

                    preview_clean = preview_clean.detach().cpu()
                    preview_degraded = preview_degraded.detach().cpu()
                    preview_pred = preview_pred.detach().cpu()

                    images = []
                    titles = []

                    for i in range(preview_clean.shape[0]):
                        images.extend(
                            [
                                preview_clean[i],
                                preview_degraded[i],
                                preview_pred[i],
                            ]
                        )

                        titles.extend(
                            [
                                f"Clean {i + 1}",
                                f"Degraded {i + 1}",
                                f"HQSNet {i + 1}",
                            ]
                        )

                    plot(
                        *images,
                        titles=titles,
                    )

            if mean_validation_loss < best_validation_loss:
                best_validation_loss = mean_validation_loss
                best_path = checkpoint_path.replace(".pth", "_best.pth")

                torch.save(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                        "best_validation_loss": best_validation_loss,
                    },
                    best_path,
                )

                print(f"Saved best checkpoint to: {best_path}")

            torch.save(
                {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best_validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )

        return history