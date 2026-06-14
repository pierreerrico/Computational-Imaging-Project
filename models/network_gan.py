from copy import deepcopy
import os
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from tqdm import tqdm
from utilities.plotter import plot

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
