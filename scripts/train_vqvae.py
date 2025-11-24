from pathlib import Path

import torch
import tqdm

from src.utils import get_device, get_mnist_dataloader
from src.vqvae import VARTokenizer

if __name__ == "__main__":
    device = get_device()

    var_tokenizer = VARTokenizer(
        scales=(1, 2, 4, 8),
        latent_dim=64,
        codebook_size=4,
    ).to(device)

    train_dl = get_mnist_dataloader(
        data_dir=Path("data"),
        batch_size=32,
        resize_size=32,
    )

    optimizer = torch.optim.AdamW(var_tokenizer.parameters(), lr=5e-5)

    for epoch in tqdm.trange(1, desc="Training"):
        with tqdm(train_dl, unit="batch") as pbar:
            for x, _ in pbar:
                x = x.to(device)

                pbar.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                x_hat, q_loss, f_loss, r_loss, scale_indices = var_tokenizer(x)
                loss = q_loss + (0.1 * f_loss) + r_loss

                pbar.set_postfix(loss=loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(var_tokenizer.parameters(), max_norm=1)
                optimizer.step()
