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


class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, rank: int = 1, alpha: float = 1.0):
        super().__init__()
        self.base = base_conv
        self.rank = rank
        self.scale = alpha / max(1, rank)

        self.down = nn.Conv2d(base_conv.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, base_conv.out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.up(self.down(x))


class LoRAConv1d(nn.Module):
    def __init__(self, base_conv: nn.Conv1d, rank: int = 1, alpha: float = 1.0):
        super().__init__()
        self.base = base_conv
        self.rank = rank
        self.scale = alpha / max(1, rank)

        self.down = nn.Conv1d(base_conv.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv1d(rank, base_conv.out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.up(self.down(x))


def inject_lora_modules(module: nn.Module, rank: int = 1, alpha: float = 1.0) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank=rank, alpha=alpha))
            count += 1
        elif isinstance(child, nn.Conv1d):
            setattr(module, name, LoRAConv1d(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += inject_lora_modules(child, rank=rank, alpha=alpha)
    return count


class FrozenDiffusionWithAdapters(nn.Module):
    """
    HSI adapter around a pretrained RGB diffusion U-Net.

    Parameter-efficient HSI adaptation around a pretrained RGB diffusion U-Net.

    Design:
    - core: frozen pretrained diffusion network (RGB domain)
    - spectral_in: lightweight 1x1 projection from HSI channels -> 3 channels
    - spectral_out: lightweight 1x1 projection from 3 channels -> HSI channels
    - optional LoRA on core Conv layers

    This removes the previous deep CNN head/tail adapters and keeps only
    parameter-efficient spectral projections + optional LoRA.
    """

    def __init__(
        self,
        core_model: nn.Module,
        hsi_channels: int,
        freeze_core: bool = True,
        core_peft: str = "none",
        lora_rank: int = 1,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.core_model = core_model
        self.hsi_channels = hsi_channels
        self.core_peft = core_peft

        if core_peft == "lora":
            n = inject_lora_modules(self.core_model, rank=lora_rank, alpha=lora_alpha)
            print(f"[LoRA] injected rank={lora_rank} adapters into {n} Conv layers in diffusion core.")

        # lightweight spectral adapters (parameter-efficient, no deep CNN head/tail)
        self.spectral_in = nn.Conv2d(hsi_channels, 3, kernel_size=1, bias=False)
        self.spectral_out = nn.Conv2d(3, hsi_channels, kernel_size=1, bias=False)

        # Stable initialization: approximate channel averaging on input
        # and channel replication on output.
        nn.init.constant_(self.spectral_in.weight, 1.0 / float(hsi_channels))
        nn.init.constant_(self.spectral_out.weight, 1.0 / 3.0)

        if freeze_core:
            self.freeze_core_model()

    def freeze_core_model(self):
        self.core_model.eval()
        for p in self.core_model.parameters():
            p.requires_grad = False

        if self.core_peft == "lora":
            for name, p in self.core_model.named_parameters():
                if "down.weight" in name or "up.weight" in name:
                    p.requires_grad = True

    def trainable_parameters(self):
        params = list(self.spectral_in.parameters()) + list(self.spectral_out.parameters())
        for p in self.core_model.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        rgb_in = self.spectral_in(x)
        rgb_out = self.core_model(rgb_in, t, **kwargs)

        if rgb_out.shape[1] == 6:
            eps_rgb, var_rgb = torch.chunk(rgb_out, 2, dim=1)
            eps_hsi = self.spectral_out(eps_rgb)
            var_hsi = self.spectral_out(var_rgb)
            return torch.cat([eps_hsi, var_hsi], dim=1)

        return self.spectral_out(rgb_out)


def build_hsi_adapter_model(
    core_model: nn.Module,
    hsi_channels: int,
    freeze_core: bool = True,
    core_peft: str = "none",
    lora_rank: int = 1,
    lora_alpha: float = 1.0,
) -> FrozenDiffusionWithAdapters:
    return FrozenDiffusionWithAdapters(
        core_model=core_model,
        hsi_channels=hsi_channels,
        freeze_core=freeze_core,
        core_peft=core_peft,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
