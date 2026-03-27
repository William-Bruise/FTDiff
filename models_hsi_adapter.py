import torch
import torch.nn as nn


class ConvHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvTail(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, out_channels: int = 31):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenDiffusionWithAdapters(nn.Module):
    """
    HSI adapter around a pretrained RGB diffusion U-Net.

    - head: HSI -> RGB
    - core: frozen pretrained diffusion network (RGB domain)
    - tail: RGB -> HSI
    """

    def __init__(
        self,
        core_model: nn.Module,
        hsi_channels: int,
        adapter_hidden_channels: int = 64,
        freeze_core: bool = True,
    ):
        super().__init__()
        self.core_model = core_model
        self.hsi_channels = hsi_channels

        self.head = ConvHead(
            in_channels=hsi_channels,
            hidden_channels=adapter_hidden_channels,
            out_channels=3,
        )
        self.tail = ConvTail(
            in_channels=3,
            hidden_channels=adapter_hidden_channels,
            out_channels=hsi_channels,
        )

        if freeze_core:
            self.freeze_core_model()

    def freeze_core_model(self):
        self.core_model.eval()
        for p in self.core_model.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return list(self.head.parameters()) + list(self.tail.parameters())

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        rgb_in = self.head(x)
        rgb_out = self.core_model(rgb_in, t, **kwargs)

        if rgb_out.shape[1] == 6:
            eps_rgb, var_rgb = torch.chunk(rgb_out, 2, dim=1)
            eps_hsi = self.tail(eps_rgb)
            var_hsi = self.tail(var_rgb)
            return torch.cat([eps_hsi, var_hsi], dim=1)

        return self.tail(rgb_out)


def build_hsi_adapter_model(
    core_model: nn.Module,
    hsi_channels: int,
    adapter_hidden_channels: int,
    freeze_core: bool = True,
) -> FrozenDiffusionWithAdapters:
    return FrozenDiffusionWithAdapters(
        core_model=core_model,
        hsi_channels=hsi_channels,
        adapter_hidden_channels=adapter_hidden_channels,
        freeze_core=freeze_core,
    )
