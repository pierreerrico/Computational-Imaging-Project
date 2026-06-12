from dataclasses import dataclass
import os
import torch
from torch.cuda import device
from IPPy.utilities.metrics import PSNR, SSIM, RE
from IPPy import operators as op, solvers as sol
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import copy as cp
from copy import deepcopy
from degradation import DegradationParameters, ImageDegradation
from focal_frequency_loss import FocalFrequencyLoss
from utils import plot # type: ignore
from typing import Any, Mapping, cast

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

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(
            self, 
            channels: int, 
            dw_expand: int = 2, 
            ffn_expand: int = 2
            ):
        super().__init__()

        groups = min(16, channels)
        while channels % groups != 0:
            groups -= 1

        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = nn.GroupNorm(groups, channels)

        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
        )
        self.gate1 = SimpleGate()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
        )

        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        self.norm2 = nn.GroupNorm(groups, channels)

        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.depthwise(h)
        h = self.gate1(h)
        h = h * self.channel_attention(h)
        h = self.conv2(h)

        y = x + self.beta * h

        h = self.norm2(y)
        h = self.conv3(h)
        h = self.gate2(h)
        h = self.conv4(h)

        return y + self.gamma * h
    
class NAFNet(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int] = (3, 256, 256),
        base_channels: int = 32,
        enc_blocks: list[int] = [1, 2, 3, 4],
        dec_blocks: list[int] = [2, 2, 2, 1],
        middle_blocks: int = 6,
    ):
        super().__init__()

        in_ch, height, width = image_shape
        out_ch = in_ch

        if len(enc_blocks) != len(dec_blocks):
            raise ValueError("enc_blocks and dec_blocks must have the same length.")

        depth = len(enc_blocks)

        if height % (2 ** depth) != 0 or width % (2 ** depth) != 0:
            raise ValueError(
                f"Image size {(height, width)} must be divisible by {2 ** depth}."
            )

        self.intro = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)

        channels = base_channels

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        for n_blocks in enc_blocks:
            self.encoders.append(
                nn.Sequential(*[
                    NAFBlock(channels)
                    for _ in range(n_blocks)
                ])
            )

            self.downs.append(
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=2,
                    stride=2,
                )
            )

            channels *= 2

        self.middle = nn.Sequential(*[
            NAFBlock(channels)
            for _ in range(middle_blocks)
        ])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for n_blocks in reversed(dec_blocks):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=1),
                    nn.PixelShuffle(2),
                )
            )

            channels //= 2

            self.decoders.append(
                nn.Sequential(*[
                    NAFBlock(channels)
                    for _ in range(n_blocks)
                ])
            )

        self.ending = nn.Conv2d(base_channels, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.intro(x)

        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)

        return residual + x

class NAFNetTrainer:
    """
    Training wrapper for a NAFNet image restoration model.

    The class handles the full supervised training loop for a degradation-based
    restoration task:

        clean image -> degradation -> degraded image -> NAFNet -> restored image

    The model is trained to reconstruct the original clean image from its
    degraded version using MAE loss.

    Features
    --------
    - Training and validation loops.
    - Automatic checkpoint saving.
    - Resume from checkpoint.
    - Best checkpoint saving based on validation loss.
    - ReduceLROnPlateau learning-rate scheduling.
    - Mixed precision training with torch.amp.
    - Gradient clipping.
    - Validation PSNR and SSIM tracking.

    Parameters
    ----------
    model : NAFNet
        The NAFNet model to train.
    """
    def __init__(self, model: NAFNet):
        self.model = model

    def train_model(
        self,
        n_epochs: int = 50,
        train_dataset: Dataset | None = None,
        validation_dataset: Dataset | None = None,
        train_degradation: ImageDegradation | None = None,
        validation_degradations: list[ImageDegradation] | None = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        checkpoint_path: str = "./weights/NAFNet/NAF_checkpoint.pth",
        resume: bool = True,
        preview_every: int | None = 5,
        preview_n: int | None = 4,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> dict[str, list[float]]:
        """
        Train the NAFNet model.

        Parameters
        ----------
        n_epochs : int
            Total number of epochs to train for.

        train_dataset : Dataset | None
            Dataset containing clean training images. Each sample is expected to
            be a tensor with shape [C, H, W].

        validation_dataset : Dataset | None
            Dataset containing clean validation images. Each sample is expected
            to be a tensor with shape [C, H, W].

        degradation : ImageDegradation | None
            Degradation object used to generate degraded inputs from clean
            images. If None, a default ImageDegradation instance is created.

        batch_size : int
            Number of images per batch.

        learning_rate : float
            Initial learning rate for AdamW.

        checkpoint_path : str
            Path where the latest training checkpoint is saved. A second
            checkpoint with suffix "_best.pth" is saved whenever validation loss
            improves.

        resume : bool
            If True and checkpoint_path exists, resume training from the saved
            checkpoint.

        preview_every : int | None
            Interval, in epochs, between validation previews. If None, previews are disabled.

        preview_n : int | None
            Number of validation samples shown in each preview.

        device : torch.device | str
            Device used for training, usually "cuda" or "cpu".

        Returns
        -------
        dict[str, list[float]]
            Training history containing:
            - train_loss
            - validation_loss
            - validation_psnr
            - validation_ssim
            - learning_rate
        """
        device = torch.device(device)
        self.model.to(device)

        if train_dataset is None:
            raise ValueError("Training dataset must be defined.")

        if validation_dataset is None:
            raise ValueError("Validation dataset must be defined.")

        if train_degradation is None:
            train_degradation = ImageDegradation()
        
        if validation_degradations is None:
            validation_degradations = [
                ImageDegradation(DegradationParameters(noise_levels=[0.005])),
                ImageDegradation(DegradationParameters(noise_levels=[0.01])),
                ImageDegradation(DegradationParameters(noise_levels=[0.05])),
                ImageDegradation(DegradationParameters(noise_levels=[0.1])),
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

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

        use_amp = device.type == "cuda"

        scaler = torch.amp.GradScaler( # type: ignore
            "cuda",
            enabled=use_amp,
        )

        history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_psnr": [],
            "validation_ssim": [],
            "learning_rate": [],
        }

        loss_mae = nn.L1Loss()
        loss_fourier = FocalFrequencyLoss()

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

            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])

            history = checkpoint["history"]
            history.setdefault("learning_rate", [])

            best_validation_loss = checkpoint["best_validation_loss"]
            start_epoch = checkpoint["epoch"] + 1

            print(f"Resumed NAFNet training from epoch {start_epoch}")

        for epoch in range(start_epoch, n_epochs):
            self.model.train()

            train_losses = []

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{n_epochs}",
                leave=True,
            )

            for clean in progress_bar:
                clean = clean.to(device, non_blocking=True)
                degraded = train_degradation(clean)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast( # type: ignore
                    "cuda",
                    enabled=use_amp,
                ):
                    pred = self.model(degraded)
                    pred = pred.clamp(0.0, 1.0)
                    loss = (
                        loss_mae(pred, clean)
                        + 0.05 * loss_fourier(pred, clean)
                        + 0.2 * (1.0 - SSIM(pred, clean))
                    )

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

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
                    for clean in validation_loader:
                        clean = clean.to(device, non_blocking=True)
                        degraded = validation_degradation(clean)

                        with torch.amp.autocast(  # type: ignore
                            "cuda",
                            enabled=use_amp,
                        ):
                            pred = self.model(degraded)
                            pred = pred.clamp(0.0, 1.0)

                            val_loss = (
                                loss_mae(pred, clean)
                                + 0.05 * loss_fourier(pred, clean)
                                + 0.2 * (1.0 - SSIM(pred, clean))
                            )

                            psnr_value = PSNR(pred.float(), clean.float())
                            ssim_value = SSIM(pred.float(), clean.float())

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

            if (
                preview_every is not None
                and preview_n is not None
                and (epoch + 1) % preview_every == 0
            ):
                self.model.eval()

                preview_clean = next(iter(validation_loader))
                preview_clean = preview_clean[:preview_n].to(device, non_blocking=True)

                with torch.no_grad():
                    preview_degraded = validation_degradations[0](preview_clean)

                    with torch.amp.autocast(  # type: ignore
                        "cuda",
                        enabled=use_amp,
                    ):
                        preview_pred = self.model(preview_degraded)
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
                            f"Restored {i + 1}",
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
                        "scaler": scaler.state_dict(),
                        "history": history,
                        "best_validation_loss": best_validation_loss,
                    },
                    best_path,
                )

                print(f"Saved best checkpoint to: {best_path}")

            checkpoint = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_validation_loss": best_validation_loss,
            }

            torch.save(checkpoint, checkpoint_path)

        return history

class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        in_groups = min(16, in_ch)
        while in_ch % in_groups != 0:
            in_groups -= 1

        out_groups = min(16, out_ch)
        while out_ch % out_groups != 0:
            out_groups -= 1

        self.main = nn.Sequential(
            nn.GroupNorm(in_groups, in_ch),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(out_groups, out_ch),
            nn.SiLU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(out_groups, out_ch),
            nn.SiLU(),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)
    
class Generator(nn.Module):
    """
    DCGAN-style generator.

    The generator maps a latent vector z to an RGB image:

        z -> G(z) -> x_fake

    Internally, the latent vector is first projected with a Linear layer into
    a low-resolution feature tensor. Then, a stack of transposed convolutions
    progressively upsamples that tensor until the target image resolution is
    reached.

    Parameters
    ----------
    img_size : tuple[int, int, int]
        Image size in PyTorch format: (channels, height, width).
        Height and width must be divisible by 16.

    latent_dim : int
        Dimension of the latent vector z.

    base_channels : int
        Base number of convolutional feature maps.

    Input
    -----
    z : torch.Tensor
        Latent tensor with shape [B, latent_dim].

    Output
    ------
    torch.Tensor
        Generated image with shape [B, C, H, W] and values in [0, 1].
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 256,
        image_shape: tuple[int, int, int] = (3, 256, 256)
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.base_channels = base_channels

        out_ch, height, width = image_shape

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Image height and width must be divisible by 16.")
        
        if base_channels % 16 != 0:
            raise ValueError(
                "Number of base channels must be divisible by 16."
            )

        self.out_ch = out_ch
        self.initial_height = height // 16
        self.initial_width = width // 16

        self.latent_to_features = nn.Sequential(
            nn.Linear(
                latent_dim,
                base_channels * self.initial_height * self.initial_width,
            ),
            nn.ReLU(inplace=True),
        )

        final_channels = base_channels // 16

        final_groups = min(16, final_channels)
        while final_channels % final_groups != 0:
            final_groups -= 1

        self.features_to_image = nn.Sequential(
            GeneratorResidualBlock(
                base_channels, 
                base_channels // 2
            ),
            GeneratorResidualBlock(
                base_channels // 2, 
                base_channels // 4
            ),
            GeneratorResidualBlock(
                base_channels // 4, 
                base_channels // 8
            ),
            GeneratorResidualBlock(
                base_channels // 8, 
                base_channels // 16
            ),

            nn.GroupNorm(
                final_groups,
                final_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                final_channels,
                out_ch, 
                kernel_size=3, 
                padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.latent_to_features(z)

        x = x.view(
            z.shape[0],
            self.base_channels,
            self.initial_height,
            self.initial_width,
        )

        return self.features_to_image(x)

    def sample_latent(
        self,
        num_samples: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            num_samples,
            self.latent_dim,
            device=device,
        )

class CriticResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.main = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AvgPool2d(2),
        )

        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                )
            ),
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)

class Critic(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        image_shape: tuple[int, int, int] = (3, 256, 256),
    ):
        super().__init__()

        in_ch, height, width = image_shape

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "Image height and width must be divisible by 16."
            )

        self.base_channels = base_channels
        self.in_ch = in_ch
        self.final_height = height // 16
        self.final_width = width // 16

        self.image_to_features = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_ch,
                    base_channels,
                    kernel_size=3,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),

            CriticResidualBlock(
                base_channels,
                base_channels * 2,
            ),
            CriticResidualBlock(
                base_channels * 2,
                base_channels * 4,
            ),
            CriticResidualBlock(
                base_channels * 4,
                base_channels * 8,
            ),
            CriticResidualBlock(
                base_channels * 8,
                base_channels * 16,
            ),
        )

        self.features_to_score = spectral_norm(
            nn.Linear(
                base_channels
                * 16
                * self.final_height
                * self.final_width,
                1,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.image_to_features(x)

        h = h.flatten(start_dim=1)

        score = self.features_to_score(h)

        return score.view(-1)

class GAN:
    """
    Generative Adversarial Network with hinge-loss critic, R1 regularization,
    and exponential moving average of the generator weights.

    The generator maps latent vectors to synthetic images. The critic assigns
    realism scores to real and generated images. During training, the critic is
    optimized with hinge loss and optional R1 regularization, while the generator
    is optimized to maximize the critic score of generated samples.

    An EMA copy of the generator is maintained throughout training and saved as
    the final generator checkpoint.

    Parameters
    ----------
    generator : Generator
        Generator network.

    critic : Critic
        Critic network.
    """

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
    ) -> None:
        self.G = generator
        self.C = critic

    def train_model(
        self,
        n_epochs: int = 50,
        lr_G: float = 1e-4,
        lr_C: float = 2e-4,
        train_dataset: Dataset | None = None,
        batch_size: int = 32,
        r1_weight: float = 5.0,
        r1_every: int = 16,
        ema_decay: float = 0.999,
        g_path: str = "./weights/GAN/GAN_G.pth",
        c_path: str = "./weights/GAN/GAN_C.pth",
        checkpoint_path: str = "./weights/GAN/GAN_checkpoint.pth",
        resume: bool = True,
        preview_every: int | None = 5,
        preview_n: int | None = 4,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> tuple[list[float], list[float]]:
        """
        Train the GAN.

        Parameters
        ----------
        n_epochs : int
            Total number of epochs.

        lr_G : float
            Generator learning rate.

        lr_C : float
            Critic learning rate.

        train_dataset : Dataset | None
            Dataset of real images. Each sample must be a tensor with shape
            [C, H, W].

        batch_size : int
            Number of images per batch.

        r1_weight : float
            Weight of the R1 gradient penalty applied to real images.

        r1_every : int
            Apply R1 regularization every `r1_every` critic steps.

        ema_decay : float
            Decay factor used to update the EMA generator.

        g_path : str
            Path where the final EMA generator weights are saved.

        c_path : str
            Path where the final critic weights are saved.

        checkpoint_path : str
            Path where the full training checkpoint is saved.

        resume : bool
            If True, resume from `checkpoint_path` when available.

        preview_every : int | None
            Show generated previews every `preview_every` epochs.
            If None, previews are disabled.

        preview_n : int | None
            Number of generated preview images.

        device : torch.device | str
            Training device.

        Returns
        -------
        tuple[list[float], list[float]]
            Pair `(g_history, c_history)` containing average generator and
            critic losses for each completed epoch.
        """

        if train_dataset is None:
            raise ValueError("Training dataset must be defined.")

        device = torch.device(device)
        use_amp = device.type == "cuda"

        torch.backends.cudnn.benchmark = True

        self.G.to(device)
        self.C.to(device)

        opt_G = torch.optim.AdamW(
            self.G.parameters(),
            lr=lr_G,
            betas=(0.0, 0.99),
        )

        opt_C = torch.optim.AdamW(
            self.C.parameters(),
            lr=lr_C,
            betas=(0.0, 0.99),
        )

        scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore
        scaler_C = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        @torch.no_grad()
        def update_ema(
            ema_model: nn.Module,
            model: nn.Module,
            decay: float,
        ) -> None:
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.mul_(decay).add_(param, alpha=1.0 - decay)

            for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)

        g_history: list[float] = []
        c_history: list[float] = []

        G_ema = deepcopy(self.G).to(device)
        G_ema.eval()

        for param in G_ema.parameters():
            param.requires_grad_(False)

        start_epoch = 0

        if resume and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            self.G.load_state_dict(checkpoint["G"])
            self.C.load_state_dict(checkpoint["C"])
            G_ema.load_state_dict(checkpoint["G_ema"])

            opt_G.load_state_dict(checkpoint["opt_G"])
            opt_C.load_state_dict(checkpoint["opt_C"])

            if "scaler_G" in checkpoint:
                scaler_G.load_state_dict(checkpoint["scaler_G"])

            if "scaler_C" in checkpoint:
                scaler_C.load_state_dict(checkpoint["scaler_C"])

            g_history = checkpoint["g_history"]
            c_history = checkpoint["c_history"]
            start_epoch = checkpoint["epoch"] + 1

            print(f"Resumed GAN training from epoch {start_epoch}")

        for epoch in range(start_epoch, n_epochs):
            self.G.train()
            self.C.train()

            g_epoch = 0.0
            c_epoch = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"GAN epoch {epoch + 1}/{n_epochs}",
                leave=True,
            )

            for step, x_real in enumerate(progress_bar, start=1):
                x_real = x_real.to(device, non_blocking=True)
                current_batch_size = x_real.shape[0]

                # ======================
                # Train critic
                # ======================
                opt_C.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore
                    z = self.G.sample_latent(
                        current_batch_size,
                        device=device,
                    )

                    x_fake = self.G(z)

                    c_real: torch.Tensor = self.C(x_real)
                    c_fake: torch.Tensor = self.C(x_fake.detach())

                    c_loss: torch.Tensor = (
                        F.relu(1.0 - c_real).mean()
                        + F.relu(1.0 + c_fake).mean()
                    )

                if step % r1_every == 0:
                    with torch.amp.autocast("cuda", enabled=False):  # type: ignore
                        x_real_reg = x_real.detach().float().requires_grad_(True)
                        c_real_reg: torch.Tensor = self.C(x_real_reg)

                        grad_real = torch.autograd.grad(
                            outputs=c_real_reg.sum(),
                            inputs=x_real_reg,
                            create_graph=True,
                        )[0]

                        r1_penalty = (
                            grad_real
                            .square()
                            .reshape(current_batch_size, -1)
                            .sum(dim=1)
                            .mean()
                        )

                        c_loss = c_loss + 0.5 * r1_weight * r1_penalty

                scaler_C.scale(c_loss).backward()
                scaler_C.step(opt_C)
                scaler_C.update()

                # ======================
                # Train generator
                # ======================
                opt_G.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore
                    z = self.G.sample_latent(
                        current_batch_size,
                        device=device,
                    )

                    x_fake = self.G(z)
                    c_fake_for_g: torch.Tensor = self.C(x_fake)
                    g_loss: torch.Tensor = -c_fake_for_g.mean()

                scaler_G.scale(g_loss).backward()
                scaler_G.step(opt_G)
                scaler_G.update()

                update_ema(G_ema, self.G, decay=ema_decay)

                g_epoch += g_loss.item()
                c_epoch += c_loss.item()

                progress_bar.set_postfix(
                    g_loss=f"{g_loss.item():.5f}",
                    c_loss=f"{c_loss.item():.5f}",
                )

            mean_g_loss = g_epoch / len(train_loader)
            mean_c_loss = c_epoch / len(train_loader)

            g_history.append(mean_g_loss)
            c_history.append(mean_c_loss)

            print(
                f"GAN epoch {epoch + 1}/{n_epochs} | "
                f"G_loss: {mean_g_loss:.6f} | "
                f"C_loss: {mean_c_loss:.6f}"
            )

            if (
                preview_every is not None
                and preview_n is not None
                and (epoch + 1) % preview_every == 0
            ):
                G_ema.eval()

                z = self.G.sample_latent(
                    preview_n,
                    device=device,
                )

                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore
                        preview_images = G_ema(z)

                    preview_images = preview_images.detach().cpu().float().clamp(0, 1)

                plot(
                    *preview_images,
                    titles=[
                        f"Epoch {epoch + 1} - sample {i + 1}"
                        for i in range(preview_n)
                    ],
                )

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "G": self.G.state_dict(),
                    "C": self.C.state_dict(),
                    "G_ema": G_ema.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_C": opt_C.state_dict(),
                    "scaler_G": scaler_G.state_dict(),
                    "scaler_C": scaler_C.state_dict(),
                    "g_history": g_history,
                    "c_history": c_history,
                },
                checkpoint_path,
            )

        os.makedirs(os.path.dirname(g_path), exist_ok=True)
        os.makedirs(os.path.dirname(c_path), exist_ok=True)

        torch.save(G_ema.state_dict(), g_path)
        torch.save(self.C.state_dict(), c_path)

        return g_history, c_history

class HQSCG:
    def __init__(self, K: op.Operator):
        self.K = K

    def batch_dot(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(a * b, dim=(1, 2, 3), keepdim=True)

    def apply_normal_operator(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        return self.K.T(self.K(x)) + mu * x

    def __call__(
        self,
        y_delta: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor,
        starting_point: torch.Tensor | None = None,
        maxiter: int = 10,
        tol: float = 1e-6,
    ) -> torch.Tensor:

        y_delta = y_delta.float()
        z = z.float()

        mu = mu.float().view(1, 1, 1, 1)

        if starting_point is None:
            x = z.clone()
        else:
            x = starting_point.float().clone()

        b = self.K.T(y_delta) + mu * z

        r = b - self.apply_normal_operator(x, mu)
        p = r.clone()

        rs_old = self.batch_dot(r, r)
        rs_initial = rs_old.clone()

        eps = 1e-8

        for _ in range(maxiter):
            Ap = self.apply_normal_operator(p, mu)

            alpha = rs_old / (self.batch_dot(p, Ap) + eps)

            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = self.batch_dot(r, r)

            relative_residual = torch.sqrt(rs_new / (rs_initial + eps))

            if relative_residual.mean() < tol:
                break

            beta = rs_new / (rs_old + eps)

            p = r + beta * p
            rs_old = rs_new

        return x
    
class HQSLayer(nn.Module):
    def __init__(
        self,
        K,
        denoiser: NAFNet,
        initial_mu: float = 1.0,
        cg_iters: int = 10,
        freeze_denoiser: bool = True,
    ) -> None:
        super().__init__()

        self.K = K
        self.denoiser = denoiser
        self.cg_iters = cg_iters
        self.freeze_denoiser = freeze_denoiser
        self.cg_solver = HQSCG(K)

        self.raw_mu = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(initial_mu)))
        )

    def get_mu(self) -> torch.Tensor:
        return F.softplus(self.raw_mu) + 1e-8

    def forward(
        self,
        y_delta: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        original_dtype = x.dtype

        x_for_denoiser = x.clamp(0.0, 1.0)

        if self.freeze_denoiser:
            with torch.no_grad():
                z = self.denoiser(x_for_denoiser)
        else:
            z = self.denoiser(x_for_denoiser)

        z = z.clamp(0.0, 1.0)

        mu = self.get_mu()

        with torch.amp.autocast("cuda", enabled=False):  # type: ignore
            x = self.cg_solver(
                y_delta=y_delta.float(),
                z=z.float(),
                mu=mu.float(),
                starting_point=x.float(),
                maxiter=self.cg_iters,
            )

        return x.to(dtype=original_dtype), z

class HQSNet(nn.Module):
    def __init__(
        self,
        K,
        denoiser: NAFNet,
        checkpoint_path: str = "./weights/NAFNet/NAFImgDenoise.pth",
        n_layers: int = 5,
        cg_iters: int = 10,
        initial_mu: float = 0.1,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        if n_layers <= 0:
            raise ValueError("n_layers must be greater than 0.")

        self.K = K
        self.device = torch.device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError("The loaded checkpoint must be a dictionary.")

        state_dict = cast(Mapping[str, Any], checkpoint["model"])

        denoiser.load_state_dict(state_dict)
        denoiser.to(self.device)
        denoiser.eval()

        for p in denoiser.parameters():
            p.requires_grad = False

        self.denoiser = denoiser

        self.layers = nn.ModuleList(
            [
                HQSLayer(
                    K=K,
                    denoiser=self.denoiser,
                    cg_iters=cg_iters,
                    initial_mu=initial_mu,
                    freeze_denoiser=True,
                )
                for _ in range(n_layers)
            ]
        )

    def initial_reconstruction(
        self,
        y_delta: torch.Tensor,
    ) -> torch.Tensor:
        return self.K.T(y_delta).clamp(0.0, 1.0)

    def forward(
        self,
        y_delta: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:

        device = next(self.parameters()).device
        y_delta = y_delta.to(device)

        x = self.initial_reconstruction(y_delta)

        intermediates: dict[str, list[torch.Tensor]] = {
            "x": [],
            "z": [],
            "mu": [],
        }

        for layer in self.layers:
            hqs_layer = cast(HQSLayer, layer)

            x, z = hqs_layer(y_delta, x)

            if return_intermediates:
                intermediates["x"].append(x)
                intermediates["z"].append(z)
                intermediates["mu"].append(hqs_layer.get_mu())

        x = x.clamp(0.0, 1.0)

        if return_intermediates:
            return x, intermediates

        return x
    
class HQSNetModel:
    """
    Training wrapper for an unrolled HQS restoration network.

    The model uses a frozen pretrained NAFNet as denoising prior and learns
    the HQS parameters, mainly one mu per unrolled layer.

    Training pipeline:

        clean image -> degradation -> degraded image -> HQSNet -> restored image
    """

    def __init__(self, model: HQSNet) -> None:
        self.model = model

    def train_model(
        self,
        n_epochs: int = 50,
        train_dataset: Dataset | None = None,
        validation_dataset: Dataset | None = None,
        train_degradation: ImageDegradation | None = None,
        validation_degradations: list[ImageDegradation] | None = None,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        checkpoint_path: str = "./weights/HQSNet/HQS_checkpoint.pth",
        resume: bool = True,
        preview_every: int | None = 5,
        preview_n: int | None = 4,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> dict[str, list[float]]:

        device = torch.device(device)
        self.model.to(device)

        if train_dataset is None:
            raise ValueError("Training dataset must be defined.")

        if validation_dataset is None:
            raise ValueError("Validation dataset must be defined.")

        if train_degradation is None:
            train_degradation = ImageDegradation()

        if validation_degradations is None:
            validation_degradations = [
                ImageDegradation(DegradationParameters(noise_levels=[0.005])),
                ImageDegradation(DegradationParameters(noise_levels=[0.01])),
                ImageDegradation(DegradationParameters(noise_levels=[0.05])),
                ImageDegradation(DegradationParameters(noise_levels=[0.1])),
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

        trainable_params = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            trainable_params,
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

        use_amp = device.type == "cuda"

        scaler = torch.amp.GradScaler( # type: ignore
            "cuda",
            enabled=use_amp,
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "validation_loss": [],
            "validation_psnr": [],
            "validation_ssim": [],
            "learning_rate": [],
        }

        loss_mae = nn.L1Loss()
        loss_fourier = FocalFrequencyLoss()

        best_validation_loss = float("inf")
        start_epoch = 0

        if resume and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            if not isinstance(checkpoint, dict):
                raise TypeError("The loaded checkpoint must be a dictionary.")

            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])

            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])

            history = checkpoint["history"]
            history.setdefault("learning_rate", [])

            best_validation_loss = checkpoint["best_validation_loss"]
            start_epoch = checkpoint["epoch"] + 1

            print(f"Resumed HQSNet training from epoch {start_epoch}")

        for epoch in range(start_epoch, n_epochs):
            self.model.train()

            # Keep frozen NAFNet in eval mode even when HQSNet is in train mode.
            self.model.denoiser.eval()

            train_losses = []

            progress_bar = tqdm(
                train_loader,
                desc=f"HQS epoch {epoch + 1}/{n_epochs}",
                leave=True,
            )

            for clean in progress_bar:
                clean = clean.to(device, non_blocking=True)
                degraded = train_degradation(clean).to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast( # type: ignore
                    "cuda",
                    enabled=use_amp,
                ):
                    pred = self.model(degraded)
                    pred = pred.clamp(0.0, 1.0)

                    loss = (
                        loss_mae(pred, clean)
                        + 0.05 * loss_fourier(pred, clean)
                        + 0.2 * (1.0 - SSIM(pred, clean))
                    )

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

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
                    for clean in validation_loader:
                        clean = clean.to(device, non_blocking=True)
                        degraded = validation_degradation(clean).to(device)

                        with torch.amp.autocast( # type: ignore
                            "cuda",
                            enabled=use_amp,
                        ):
                            pred = self.model(degraded)
                            pred = pred.clamp(0.0, 1.0)

                            val_loss = (
                                loss_mae(pred, clean)
                                + 0.05 * loss_fourier(pred, clean)
                                + 0.2 * (1.0 - SSIM(pred, clean))
                            )

                            psnr_value = PSNR(pred.float(), clean.float())
                            ssim_value = SSIM(pred.float(), clean.float())

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

            mus = [
                layer.get_mu().detach().item()
                for layer in self.model.layers
                if isinstance(layer, HQSLayer)
            ]

            print(
                f"HQS epoch {epoch + 1}/{n_epochs} | "
                f"train_loss: {mean_train_loss:.6f} | "
                f"val_loss: {mean_validation_loss:.6f} | "
                f"val_PSNR: {mean_validation_psnr:.4f} | "
                f"val_SSIM: {mean_validation_ssim:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"mu: {[round(m, 4) for m in mus]}"
            )

            if (
                preview_every is not None
                and preview_n is not None
                and (epoch + 1) % preview_every == 0
            ):
                preview_clean = next(iter(validation_loader))
                preview_clean = preview_clean[:preview_n].to(device, non_blocking=True)

                with torch.no_grad():
                    preview_degraded = validation_degradations[0](preview_clean)

                    with torch.amp.autocast( # type: ignore
                        "cuda",
                        enabled=use_amp,
                    ):
                        preview_pred = self.model(preview_degraded)
                        preview_pred = preview_pred.clamp(0.0, 1.0)

                preview_clean = preview_clean.detach().float().cpu()
                preview_degraded = preview_degraded.detach().float().cpu()
                preview_pred = preview_pred.detach().float().cpu()

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
                            f"HQS restored {i + 1}",
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
                        "scaler": scaler.state_dict(),
                        "history": history,
                        "best_validation_loss": best_validation_loss,
                    },
                    best_path,
                )

                print(f"Saved best HQSNet checkpoint to: {best_path}")

            torch.save(
                {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "best_validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )

        return history
    
class DRUNetDenoiser(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:

        if isinstance(sigma, float):
            sigma = torch.tensor(
                sigma,
                device=x.device,
                dtype=x.dtype,
            )

        sigma_map = torch.ones(
            x.shape[0],
            1,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        ) * sigma

        inp = torch.cat([x, sigma_map], dim=1)

        return self.model(inp)