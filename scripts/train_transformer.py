from pathlib import Path

import torch
import tqdm

from src.transformer import VARTransformer
from src.utils import get_device, get_mnist_dataloader
from src.vqvae import VARTokenizer

if __name__ == "__main__":
    var_transformer = VARTransformer(vocab_size=4)
    loss_fn = torch.nn.CrossEntropyLoss()
    t_optimizer = torch.optim.AdamW(var_transformer.parameters(), lr=3e-4)
    var_tokenizer = VARTokenizer()  # TODO: obviously load it from saved .pt
    var_tokenizer.eval()
    train_dl = get_mnist_dataloader(data_dir=Path("data"))

    device = get_device()
    var_tokenizer.to(device)
    var_transformer.to(device)

    for epoch in tqdm.trange(1, desc="Training"):
        with tqdm(train_dl, unit="batch") as pbar:
            for x, _ in pbar:
                x = x.to(device)

                pbar.set_description(f"Epoch {epoch}")
                t_optimizer.zero_grad()

                with torch.no_grad():
                    x_hat, q_loss, f_loss, r_loss, scale_indices = var_tokenizer(x)

                # concatenate along the scale dimension(scale_indices) to
                # match the transformer input
                prepared_indices = torch.cat(scale_indices, dim=1)

                output = var_transformer(scale_indices)
                loss = loss_fn(
                    output.view(-1, var_transformer.vocab_size),
                    prepared_indices.reshape(-1),
                )
                pbar.set_postfix(loss=loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(var_transformer.parameters(), max_norm=1)
                t_optimizer.step()
