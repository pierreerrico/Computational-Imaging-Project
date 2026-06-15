import os
import torch

from IPPy.utilities.metrics import PSNR, SSIM, RE
from IPPy import operators as op
from utilities.plotter import plot

from torchvision.models import resnet50, ResNet50_Weights

try:
    from pytorch_pretrained_biggan import BigGAN
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'pytorch_pretrained_biggan'. Install it with: "
        "pip install pytorch-pretrained-biggan"
    ) from exc


class DGPImageReconstruction:
    def __init__(
        self,
        model_name: str = "biggan-deep-256",
        lambda_values: list[float] | None = None,
        max_iters: int = 200,
        learning_rate: float = 5e-3,
        truncation: float = 0.7,
        print_every: int = 25,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model_name = model_name
        self.lambda_values = lambda_values or [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.truncation = truncation
        self.print_every = print_every
        self.device = torch.device(device)

        self.G = BigGAN.from_pretrained(self.model_name).to(self.device).eval()
        for p in self.G.parameters():
            p.requires_grad = False

        self.z_dim = 128
        self.c_dim = 1000

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.classifier = resnet50(weights=weights).to(self.device).eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.class_transform = weights.transforms()
        self.class_names = weights.meta["categories"]

    def estimate_class(self, image: torch.Tensor) -> tuple[int, str, float]:
        if image.ndim == 4:
            image = image[0]

        x = self.class_transform(image.detach().cpu()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=1)
            prob, idx = torch.max(probs, dim=1)

        class_index = int(idx.item())
        class_name = self.class_names[class_index]
        class_probability = float(prob.item())

        print(
            f"Estimated ImageNet class: {class_index} | "
            f"{class_name} | p={class_probability:.4f}",
            flush=True,
        )

        return class_index, class_name, class_probability

    def make_class_vector(self, class_index: int) -> torch.Tensor:
        c = torch.zeros(1, self.c_dim, device=self.device)
        c[:, class_index] = 1.0
        return c

    def generate(self, z: torch.Tensor, class_vector: torch.Tensor) -> torch.Tensor:
        x = self.G(z, class_vector, self.truncation)
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    def initialise_latent(self) -> tuple[torch.nn.Parameter, torch.Tensor]:
        z_init = torch.randn(1, self.z_dim, device=self.device)
        z = torch.nn.Parameter(z_init.detach().clone())
        return z, z_init.detach()

    def __call__(
        self,
        y_d: torch.Tensor,
        K: op.Operator,
        x_gt: torch.Tensor | None = None,
        save_dir: str | None = None,
        preview: bool = False,
    ) -> list[dict]:

        if y_d.ndim != 4:
            raise ValueError(f"Expected y_d with shape [B,C,H,W], got {y_d.shape}")

        if x_gt is not None and x_gt.shape != y_d.shape:
            raise ValueError(
                f"x_gt and y_d must have the same shape, got {x_gt.shape} and {y_d.shape}"
            )

        y_d = y_d.detach().to(self.device).float()

        if x_gt is not None:
            x_gt = x_gt.detach().to(self.device).float()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        class_index, class_name, class_probability = self.estimate_class(y_d)
        class_vector = self.make_class_vector(class_index)

        results = []

        for lambda_value in self.lambda_values:
            print(f"Running DGP reconstruction with lambda = {lambda_value:.1e}", flush=True)

            z, z_init = self.initialise_latent()

            optimizer = torch.optim.Adam(
                [z],
                lr=self.learning_rate,
            )

            losses = []
            data_losses = []
            latent_losses = []

            for iteration in range(self.max_iters):
                optimizer.zero_grad()

                x_hat = self.generate(z, class_vector)
                y_hat = K(x_hat)

                data_loss = 0.5 * torch.mean((y_hat - y_d) ** 2)
                latent_loss = 0.5 * torch.mean((z - z_init) ** 2)

                loss = data_loss + lambda_value * latent_loss

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                data_losses.append(data_loss.item())
                latent_losses.append(latent_loss.item())

                if self.print_every > 0 and (iteration + 1) % self.print_every == 0:
                    print(
                        f"iter={iteration + 1:04d} | "
                        f"loss={loss.item():.6f} | "
                        f"data={data_loss.item():.6f} | "
                        f"latent={latent_loss.item():.6f}",
                        flush=True,
                    )

            with torch.no_grad():
                x_hat = self.generate(z, class_vector)

            x_hat = x_hat.detach().clamp(0.0, 1.0)

            result = {
                "lambda": lambda_value,
                "reconstruction": x_hat,
                "z": z.detach().cpu(),
                "class_index": class_index,
                "class_name": class_name,
                "class_probability": class_probability,
                "losses": losses,
                "data_losses": data_losses,
                "latent_losses": latent_losses,
            }

            lambda_name = f"{lambda_value:.0e}".replace("-", "m")

            if x_gt is not None:
                result["psnr"] = PSNR(x_hat, x_gt)
                result["ssim"] = SSIM(x_hat, x_gt)
                result["re"] = RE(x_hat, x_gt)

                if isinstance(result["psnr"], torch.Tensor):
                    result["psnr"] = result["psnr"].item()
                if isinstance(result["ssim"], torch.Tensor):
                    result["ssim"] = result["ssim"].item()
                if isinstance(result["re"], torch.Tensor):
                    result["re"] = result["re"].item()

                print(
                    f"Done | "
                    f"PSNR={result['psnr']:.2f} dB | "
                    f"SSIM={result['ssim']:.4f} | "
                    f"RE={result['re']:.4f} | "
                    f"class={class_index} {class_name}",
                    flush=True,
                )
            else:
                print(f"Done | class={class_index} {class_name}", flush=True)

            if save_dir is not None:
                torch.save(
                    {
                        "lambda": lambda_value,
                        "reconstruction": x_hat.detach().cpu(),
                        "z": z.detach().cpu(),
                        "class_index": class_index,
                        "class_name": class_name,
                        "class_probability": class_probability,
                        "losses": losses,
                        "data_losses": data_losses,
                        "latent_losses": latent_losses,
                        "psnr": result.get("psnr"),
                        "ssim": result.get("ssim"),
                        "re": result.get("re"),
                    },
                    os.path.join(save_dir, f"dgp_lambda_{lambda_name}.pt"),
                )

            if preview:
                plot(
                    y_d[0].detach().cpu(),
                    x_hat[0].detach().cpu(),
                    *( [x_gt[0].detach().cpu()] if x_gt is not None else [] ),
                    titles=[
                        "Degraded image",
                        f"DGP reconstruction\nλ={lambda_value:.1e}\nclass={class_name}",
                        *(["Ground truth"] if x_gt is not None else []),
                    ],
                )

            results.append(result)

        return results