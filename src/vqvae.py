import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VARTokenizer(nn.Module):
    def __init__(
        self,
        scales: tuple[int] = (1, 2, 4, 8),
        latent_dim: int = 64,
        codebook_size: int = 128,
    ):
        super().__init__()
        self.scales = scales
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
        )

        conv_params = {
            "in_channels": latent_dim,
            "out_channels": latent_dim,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        }
        self.convs = nn.ModuleList([nn.Conv2d(**conv_params) for _ in self.scales])

    def forward(self, x: Tensor) -> Tensor:
        f = self.encoder(x)
        f_original = f.clone()

        f_hat = torch.zeros_like(f)
        q_loss = 0.0
        scale_indices: list[Tensor] = []

        for scale_id, scale in enumerate(self.scales):
            f_inter_scale = F.interpolate(
                f,
                size=(scale, scale),
                mode="bilinear",
                align_corners=False,
            )
            z_q, indices, loss_scale = self.quantizer(f_inter_scale)
            z_inter_full = F.interpolate(
                z_q,
                size=(self.scales[-1], self.scales[-1]),
                mode="bilinear",
                align_corners=False,
            )
            conv_out = self.convs[scale_id](z_inter_full)
            f = f - conv_out
            f_hat = f_hat + conv_out
            q_loss = q_loss + loss_scale
            scale_indices.append(indices)

        x_hat = self.decoder(f_hat)

        f_loss = (f_hat - f_original).pow(2).mean()
        r_loss = (x_hat - x).pow(2).mean()

        return x_hat, q_loss, f_loss, r_loss, scale_indices

    def tokens_to_image(self, scale_indices: list[Tensor]) -> Tensor:
        batch_size = scale_indices[0].shape[0]
        f_hat = torch.zeros(
            batch_size,
            self.encoder.latent_conv.out_channels,
            self.scales[-1],
            self.scales[-1],
        ).to(self.quantizer.table.weight.device)

        for scale_id, indices in enumerate(scale_indices):
            h = w = self.scales[scale_id]
            indices_grid = indices.view(batch_size, h, w)

            # Lookup (B, H, W, C) -> Permute (B, C, H, W)
            z_q = self.quantizer.table(indices_grid).permute(0, 3, 1, 2)

            # Interpolate to the full size
            z_inter = F.interpolate(
                z_q,
                size=(self.scales[-1], self.scales[-1]),
                mode="bilinear",
                align_corners=False,
            )

            # Pass through the appropriate convolution layer for this scale
            conv_out = self.convs[scale_id](z_inter)
            f_hat = f_hat + conv_out

        return self.decoder(f_hat)

    def get_codebook_stats(self) -> dict[str, float]:
        return {
            "mean": self.quantizer.table.weight.mean().item(),
            "sd": self.quantizer.table.weight.sd().item(),
        }


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        beta: float = 0.25,
    ):
        super().__init__()
        self.beta = beta
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.table = nn.Embedding(  # (K, C)
            num_embeddings=codebook_size,
            embedding_dim=self.embedding_dim,
        )
        self._inited = False

    def forward(self, z: Tensor) -> Tensor:
        batch_size, n_channels, height, width = z.shape
        assert n_channels == self.embedding_dim, (
            "Channel and embedding dimension is not the same"
        )

        # (B, C, H, W) -> (B, H, W, C)
        z = z.permute(0, 2, 3, 1)

        # (B, H, W, C) -> (B*H*W, C) = (N, C)
        z_flat = z.reshape(-1, n_channels)

        if not self._inited and z_flat.shape[0] > self.codebook_size:
            self._inited = True
            choices = torch.randperm(z_flat.shape[0])
            self.table.weight.data.copy_(z_flat[choices[: self.codebook_size]])

        # (N, K)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)  # (N, C) -> (N, 1)
            + self.table.weight.pow(2).sum(dim=1)  # (K, C) -> (K,)
            - 2 * (z_flat @ self.table.weight.T)  # (N, C) @ (C, K) -> (N, K)
        )

        # Indices of the closest embedding for each input embedding vector
        indices = distances.argmin(dim=1)
        z_q_raw = self.table.weight[indices]

        # Loss calculation before the straight through estimator
        codebook_loss = (z_flat.detach() - z_q_raw).pow(2).mean()
        commitment_loss = (z_flat - z_q_raw.detach()).pow(2).mean()
        total_loss = codebook_loss + (self.beta * commitment_loss)

        # Straight through estimator to pass the gradients
        z_q = z_flat + (z_q_raw - z_flat).detach()  # (N, C)

        # Reshape the quantized embeddings back to the expected shape
        z_q = z_q.view(batch_size, height, width, n_channels)
        z_q = z_q.permute(0, 3, 1, 2)

        per_back_indices = indices.view(batch_size, -1)

        return z_q, per_back_indices, total_loss


class ResBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        num_groups: int = 32,
    ):
        super().__init__()
        conv_params = {
            "in_channels": n_channels,
            "out_channels": n_channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        }
        self.conv1 = nn.Conv2d(**conv_params)
        self.conv2 = nn.Conv2d(**conv_params)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=n_channels)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=n_channels)

    def forward(self, x: Tensor) -> Tensor:
        y = self.gn1(x)
        y = F.silu(y)
        y = self.conv1(y)

        y = self.gn2(y)
        y = F.silu(y)
        y = self.conv2(y)

        return x + y


class NonLocalBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=n_channels, num_groups=32)

        conv_params = {
            "in_channels": n_channels,
            "out_channels": n_channels,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
        }
        self.q_conv = nn.Conv2d(**conv_params)
        self.k_conv = nn.Conv2d(**conv_params)
        self.v_conv = nn.Conv2d(**conv_params)
        self.out_conv = nn.Conv2d(**conv_params)

    def forward(self, x: Tensor) -> Tensor:
        y = self.gn(x)

        q = self.q_conv(y)  # (B, C, H, W) -> (B, C, H, W)
        k = self.k_conv(y)
        v = self.v_conv(y)

        q = torch.flatten(q, 2)  # (B, C, H, W) -> (B, C, H*W)
        k = torch.flatten(k, 2)
        v = torch.flatten(v, 2)

        n_channels = y.shape[1]

        q_t = q.permute(0, 2, 1)  # Transpose pixels and channels

        attn = q_t.bmm(k)  # (B, H*W, C) @ (B, C, H*W) -> (B, H*W, H*W)
        attn_norm = torch.softmax(attn.div(int(n_channels) ** 0.5), dim=-1)

        # (B, C, H*W) @ (B, H*W, H*W)
        out = v.bmm(attn_norm.permute(0, 2, 1))
        out = out.reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3])
        out = self.out_conv(out)

        return out + x


class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
    ):
        super().__init__()
        downsample_params = {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
        }
        self.initial_conv = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, **downsample_params)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, **downsample_params)

        self.resblock1 = ResBlock(n_channels=64, num_groups=32)
        self.resblock2 = ResBlock(n_channels=128, num_groups=32)

        self.resblock_end1 = ResBlock(n_channels=256, num_groups=32)
        self.nonlocal_block = NonLocalBlock(n_channels=256)
        self.resblock_end2 = ResBlock(n_channels=256, num_groups=32)

        self.gn = nn.GroupNorm(num_groups=32, num_channels=256)
        self.latent_conv = nn.Conv2d(
            in_channels=256,
            out_channels=latent_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_conv(x)

        x = self.resblock1(x)
        x = self.conv1(x)

        x = self.resblock2(x)
        x = self.conv2(x)

        x = self.resblock_end1(x)
        x = self.nonlocal_block(x)
        x = self.resblock_end2(x)

        x = self.gn(x)
        x = F.silu(x)
        x = self.latent_conv(x)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=latent_dim,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.resblock_nl1 = ResBlock(n_channels=256, num_groups=32)
        self.resblock_nl2 = ResBlock(n_channels=256, num_groups=32)

        self.nonlocal_block = NonLocalBlock(n_channels=256)

        self.upblock1 = UpsampleBlock(in_channels=256, out_channels=128)
        self.upblock2 = UpsampleBlock(in_channels=128, out_channels=64)

        self.resblock1 = ResBlock(n_channels=256, num_groups=32)
        self.resblock2 = ResBlock(n_channels=128, num_groups=32)

        self.gn = nn.GroupNorm(num_channels=64, num_groups=32)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)

        x = self.resblock_nl1(x)
        x = self.nonlocal_block(x)
        x = self.resblock_nl2(x)

        x = self.resblock1(x)
        x = self.upblock1(x)
        x = self.resblock2(x)
        x = self.upblock2(x)

        x = self.gn(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x
