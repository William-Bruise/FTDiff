import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ConvHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 3,
        num_blocks: int = 8,
    ):
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_channels) for _ in range(num_blocks)])
        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.proj(x)
        return x


class ConvTail(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 256,
        out_channels: int = 31,
        num_blocks: int = 8,
    ):
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_channels) for _ in range(num_blocks)])
        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.proj(x)
        return x


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
        adapter_hidden_channels: int = 256,
        adapter_num_blocks: int = 8,
        freeze_core: bool = True,
    ):
        super().__init__()
        self.core_model = core_model
        self.hsi_channels = hsi_channels

        self.head = ConvHead(
            in_channels=hsi_channels,
            hidden_channels=adapter_hidden_channels,
            out_channels=3,
            num_blocks=adapter_num_blocks,
        )
        self.tail = ConvTail(
            in_channels=3,
            hidden_channels=adapter_hidden_channels,
            out_channels=hsi_channels,
            num_blocks=adapter_num_blocks,
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
    adapter_num_blocks: int = 8,
    freeze_core: bool = True,
) -> FrozenDiffusionWithAdapters:
    return FrozenDiffusionWithAdapters(
        core_model=core_model,
        hsi_channels=hsi_channels,
        adapter_hidden_channels=adapter_hidden_channels,
        adapter_num_blocks=adapter_num_blocks,
        freeze_core=freeze_core,
    )
