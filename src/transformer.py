import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils import get_device


class VARTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        scales: tuple[int] = (1, 2, 4, 8),
        n_transformer_layers: int = 4,
        device=get_device(),
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.scales = scales
        self.max_sequence_length = sum([x**2 for x in self.scales])
        self.n_transformer_layers = n_transformer_layers

        attention_mask = self.mask_matrix(device)
        self.register_buffer("attention_mask", attention_mask)

        self.start_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.vocab_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.max_sequence_length,
            embedding_dim=embedding_dim,
        )

        self.blocks = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=16,
                dim_feedforward=4 * embedding_dim,
                batch_first=True,
            ),
            num_layers=self.n_transformer_layers,
        )

        self.classification_head = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.vocab_size,
        )

    @torch.no_grad()
    def mask_matrix(self, device: torch.device) -> Tensor:
        size = self.max_sequence_length
        mask = torch.zeros(size, size).to(device)
        mask = torch.fill(mask, -torch.inf)

        # First add the mask for the start token
        mask[0, 0] = 0
        current_start = 1

        # Add the mask for each of the scales
        # except the last which is not used as input
        for scale in self.scales[:-1]:
            # We are working with the upsampled scale (hence *2)
            n_tokens = (scale * 2) ** 2

            current_end = current_start + n_tokens
            mask[current_start:current_end, 0:current_end] = 0
            current_start = current_start + n_tokens

        return mask

    def forward(self, x: list[Tensor], batch_size: int | None = None) -> Tensor:
        # Each x is a list of Tensors for each scale
        # where each Tensor has shape (batch_size, N) for that scale
        batch_size = batch_size if len(x) == 0 else x[0].shape[0]
        if batch_size is None:
            raise ValueError(
                "Input list must be batched, or batch_size must be passed."
            )
        input_list = [self.start_token.expand(batch_size, -1, -1)]

        # Building the input for the transformer
        for scale_idx, scale_tokens in enumerate(x):
            if scale_idx == (len(self.scales) - 1):
                continue

            # 1. Embedding the tokens in embeddings
            # - Scale tokens is (B, seq_length)
            # - Embedding is (B, seq_length, C)
            words: Tensor = self.vocab_embeddings(scale_tokens)

            # 2. Reshape into a batched grid of embeddings for the scale (B, C, H, W)
            # - We push the sequence length to th__e end, and then reshape it to the grid
            # - This prepares us for the interpolation
            n_channels = self.embedding_dim
            height, width = self.scales[scale_idx], self.scales[scale_idx]
            words = words.permute(0, 2, 1)  # (B, seq_length, C) -> (B, C, seq_length)
            words = words.reshape(
                -1, n_channels, height, width
            )  # (B, C, seq_length) -> (B, C, H, W)

            # 3. Interpolate to the size of the next scale (H*2, W*2)
            words = F.interpolate(
                words,
                scale_factor=2,
                mode="bilinear",
            )  # (B, C, H, W)

            # 4. Reshape it back to the sequence for the transformer (B, H*2*W*2, C)
            words = words.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            words = words.reshape(
                -1, height * width * 4, n_channels
            )  # (B, H, W, C) -> (B, seq_length, C)

            input_list.append(words)

        x = torch.cat(input_list, dim=1)
        sequence_length = x.shape[1]  # (B, seq_len, C)
        pos_indices = torch.arange(
            start=0,
            end=sequence_length,
            step=1,
        ).to(x.device)
        pos_emb = self.positional_embeddings(pos_indices)

        x = x + pos_emb
        mask = self.attention_mask[:sequence_length, :sequence_length]
        output: nn.TransformerEncoder = self.blocks(x, mask=mask)
        output = self.classification_head(output)
        return output


if __name__ == "__main__":
    from loguru import logger

    device = get_device()
    x = [
        torch.arange(0, 1, 1).unsqueeze(0).to(device),
        torch.arange(0, 4, 1).unsqueeze(0).to(device),
        torch.arange(0, 16, 1).unsqueeze(0).to(device),
        torch.arange(0, 64, 1).unsqueeze(0).to(device),
    ]
    var_test = VARTransformer(32).to(device)
    y = var_test(x)
    logger.info(f"Output shape: {y.shape}")
