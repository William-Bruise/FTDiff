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


def inject_lora_modules(
    module: nn.Module,
    rank: int = 1,
    alpha: float = 1.0,
    target_conv2d: str = "1x1",
    target_conv1d: bool = True,
) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            allow = (target_conv2d == "all") or (
                target_conv2d == "1x1" and child.kernel_size == (1, 1)
            )
            if allow:
                setattr(module, name, LoRAConv2d(child, rank=rank, alpha=alpha))
                count += 1
        elif isinstance(child, nn.Conv1d):
            if target_conv1d:
                setattr(module, name, LoRAConv1d(child, rank=rank, alpha=alpha))
                count += 1
        else:
            count += inject_lora_modules(
                child,
                rank=rank,
                alpha=alpha,
                target_conv2d=target_conv2d,
                target_conv1d=target_conv1d,
            )
    return count


class FrozenDiffusionWithAdapters(nn.Module):
    """
    HSI adapter around a pretrained RGB diffusion U-Net.

    Replace pretrained RGB core I/O with trainable CNN head/tail for HSI:
    - replace first input conv in UNet with ConvHead (HSI -> core stem channels)
    - replace final output conv in UNet with ConvTail (core out channels -> HSI)
    - freeze middle of diffusion core; train only replaced head/tail
    - optional LoRA on the remaining core if requested
    """

    def __init__(
        self,
        core_model: nn.Module,
        hsi_channels: int,
        adapter_hidden_channels: int = 256,
        adapter_num_blocks: int = 8,
        freeze_core: bool = True,
        core_peft: str = "none",
        lora_rank: int = 1,
        lora_alpha: float = 1.0,
        lora_conv2d_target: str = "1x1",
        lora_enable_conv1d: bool = True,
    ):
        super().__init__()
        self.core_model = core_model
        self.hsi_channels = hsi_channels
        self.core_peft = core_peft
        self.lora_conv2d_target = lora_conv2d_target
        self.lora_enable_conv1d = lora_enable_conv1d

        self.head, self.tail = self._replace_core_io_with_cnn_adapters(
            hsi_channels=hsi_channels,
            hidden_channels=adapter_hidden_channels,
            num_blocks=adapter_num_blocks,
        )

        if core_peft == "lora":
            n = inject_lora_modules(
                self.core_model,
                rank=lora_rank,
                alpha=lora_alpha,
                target_conv2d=lora_conv2d_target,
                target_conv1d=lora_enable_conv1d,
            )
            print(
                f"[LoRA] injected rank={lora_rank} adapters into {n} Conv layers in diffusion core "
                f"(conv2d_target={lora_conv2d_target}, conv1d={lora_enable_conv1d})."
            )

        if freeze_core:
            self.freeze_core_model()

    def _replace_core_io_with_cnn_adapters(self, hsi_channels: int, hidden_channels: int, num_blocks: int):
        if not hasattr(self.core_model, "input_blocks") or not hasattr(self.core_model, "out"):
            raise AttributeError("core_model must provide `input_blocks` and `out` modules (UNetModel).")

        first_block = self.core_model.input_blocks[0]
        in_module = first_block[0]
        if not isinstance(in_module, nn.Conv2d):
            raise TypeError("Expected core_model.input_blocks[0][0] to be nn.Conv2d.")
        stem_channels = in_module.out_channels
        head = ConvHead(
            in_channels=hsi_channels,
            hidden_channels=hidden_channels,
            out_channels=stem_channels,
            num_blocks=num_blocks,
        )
        first_block[0] = head

        out_module = self.core_model.out[-1]
        if not isinstance(out_module, nn.Conv2d):
            raise TypeError("Expected core_model.out[-1] to be nn.Conv2d.")
        if out_module.out_channels % 3 != 0:
            raise ValueError(f"Expected core output channels divisible by 3, got {out_module.out_channels}.")
        out_factor = out_module.out_channels // 3  # 1 for eps; 2 for eps+var
        tail = ConvTail(
            in_channels=out_module.in_channels,
            hidden_channels=hidden_channels,
            out_channels=hsi_channels * out_factor,
            num_blocks=num_blocks,
        )
        self.core_model.out[-1] = tail
        return head, tail

    def freeze_core_model(self):
        self.core_model.train()
        for p in self.core_model.parameters():
            p.requires_grad = False

        for p in self.head.parameters():
            p.requires_grad = True
        for p in self.tail.parameters():
            p.requires_grad = True

        if self.core_peft == "lora":
            for name, p in self.core_model.named_parameters():
                if "down.weight" in name or "up.weight" in name:
                    p.requires_grad = True

    def trainable_parameters(self):
        params = list(self.head.parameters()) + list(self.tail.parameters())
        for p in self.core_model.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.core_model(x, t, **kwargs)


def build_hsi_adapter_model(
    core_model: nn.Module,
    hsi_channels: int,
    adapter_hidden_channels: int = 256,
    adapter_num_blocks: int = 8,
    freeze_core: bool = True,
    core_peft: str = "none",
    lora_rank: int = 1,
    lora_alpha: float = 1.0,
    lora_conv2d_target: str = "1x1",
    lora_enable_conv1d: bool = True,
) -> FrozenDiffusionWithAdapters:
    return FrozenDiffusionWithAdapters(
        core_model=core_model,
        hsi_channels=hsi_channels,
        adapter_hidden_channels=adapter_hidden_channels,
        adapter_num_blocks=adapter_num_blocks,
        freeze_core=freeze_core,
        core_peft=core_peft,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_conv2d_target=lora_conv2d_target,
        lora_enable_conv1d=lora_enable_conv1d,
    )
