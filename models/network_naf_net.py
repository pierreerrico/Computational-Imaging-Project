
from dataclasses import dataclass
import os
import torch
from IPPy.utilities.metrics import PSNR, SSIM
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from Project.utilities.degradation import DegradationParameters, ImageDegradation
from focal_frequency_loss import FocalFrequencyLoss
from utilities.plotter import plot

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
