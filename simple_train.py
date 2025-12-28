import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================================================================
def one_hot_encode(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


image_size = 32


class OneHotMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root='data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
            ])
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        one_hot_label = one_hot_encode(label)
        return image, one_hot_label

    def __len__(self):
        return len(self.dataset)


train_dataset = OneHotMNIST(train=True)
test_dataset = OneHotMNIST(train=False)

batch_size = 50
n_splats = 100

train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class DummyTextCond(nn.Module):
    def __init__(self, token_sequence_length, d_dim):
        super().__init__()
        self.token_sequence_length = token_sequence_length
        self.d_channels = d_dim

        self.token_proj = nn.Linear(10, token_sequence_length * d_dim)
        self.token_norm = nn.LayerNorm(d_dim)

    def forward(self, label_vector):
        # label_vector is of size [B, 10]
        b = label_vector.size(0)

        tokens = self.token_proj(label_vector)
        tokens = tokens.reshape(b, self.token_sequence_length, self.d_channels)
        tokens = self.token_norm(tokens)

        return tokens


from modules.rigsig_single_step import RIGSIG

rigsig = RIGSIG(
    num_pos_freq=5,
    num_col_freq=5,
    c_dim=1,
    d_dim=256,
    num_blocks=8,
    num_heads=8,
    base_attn_dropout=0.1,
    cross_attn_dropout=0.1,
    mlp_dropout=0.2,
).to(device)
mnist_text_embed = DummyTextCond(1, rigsig.d_dim).to(device)
lr = 5e-6
rigsig_optimizer = torch.optim.AdamW(rigsig.parameters(), lr)
te_model_optimizer = torch.optim.AdamW(mnist_text_embed.parameters(), lr)

total = sum(p.numel() for p in rigsig.parameters() if p.requires_grad)
print(f"Trainable parameters: {total:,}")


def generate_coordinates(
        image_tensor: torch.Tensor,  # [b, c, h, w] but c ignored
        jitter_std: float = 0.0,  # Gaussian std (e.g., 0.5 / min(h,w) for pixel-relative)
) -> torch.Tensor:
    b, _, h, w = image_tensor.shape
    device = image_tensor.device

    # Normalized grids: x/y from -1 to 1, center=0
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),  # Avoid exact ±1 if needed
        torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
        indexing='ij'
    )
    # Flatten to [1, h*w, 2]
    xy = torch.stack([x_grid, y_grid], dim=-1).view(1, h * w, 2)

    # z = (h - w) / max(h, w)  # 0 for square
    z = torch.full([1, h * w, 1], (h - w) / max(h, w), device=device)

    # Concat to [1, m, 3], repeat for batch
    coords = torch.cat([xy, z], dim=-1).repeat(b, 1, 1)  # [b, m, 3]

    if jitter_std > 0:
        jitter = torch.randn_like(coords) * jitter_std
        coords = coords + jitter
        coords.clamp_(-1, 1)  # Keep in cube

    return coords


import torch
import matplotlib.pyplot as plt
import math


def render_image(tensor: torch.Tensor, title: str = None):
    B, C, H, W = tensor.shape

    cols = math.ceil(math.sqrt(B))
    rows = math.ceil(B / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if B > 1 else [axes]

    for i in range(rows * cols):
        ax = axes[i]
        if i < B:
            img = tensor[i]
            if C == 1:
                img = img.squeeze(0)
                ax.imshow(img.cpu(), cmap='gray')
            elif C == 3:
                img = img.permute(1, 2, 0)
                ax.imshow(img.cpu())
            else:
                raise ValueError(f"Unsupported number of channels: {C}")
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)

    plt.tight_layout()
    plt.show()


def colors_to_image(
        colors: torch.Tensor,  # [b, m, c] from sample_col_from_splats
        h: int,  # Original height
        w: int,  # Original width (h==w for square)
) -> torch.Tensor:
    # assert colors.ndim == 3 and colors.size(1) == h * w and colors.size(2) == c_dim, "Invalid colors shape"

    # Transpose to [b, c, m] then reshape to [b, c, h, w]
    image = colors.transpose(1, 2).view(colors.size(0), colors.size(2), h, w)

    return image


num_epochs = 20

from tqdm import tqdm

for e in range(num_epochs):
    rigsig.train()
    mnist_text_embed.train()

    rigsig.zero_grad()
    mnist_text_embed.zero_grad()

    train_loss = 0
    train_losses = []

    for i, (image, label) in tqdm(enumerate(train_dloader), total=len(train_dloader), desc=f"E{e + 1} - TRAIN"):
        image, label = image.to(device), label.to(device)
        text_cond = mnist_text_embed(label)
        image_coords = generate_coordinates(image)

        pred_splats = rigsig(batch_size, n_splats, text_cond)
        sampled_colors = rigsig.splat_control.sample_col_from_splats(pred_splats, image_coords)
        if i % 100 == 0:
            sampled_image = colors_to_image(sampled_colors.detach(), 32, 32)
            render_image(sampled_image)
        target_colors = image.view(batch_size, 1, -1).transpose(1, 2)
        loss = nn.functional.mse_loss(sampled_colors, target_colors)
        train_loss += loss.item()
        train_losses.append(loss.item())
        loss.backward()
        rigsig_optimizer.step()
        te_model_optimizer.step()

    train_loss /= len(train_dloader)

    plt.title("train")
    plt.plot(train_losses)
    # plt.legend()
    plt.show()

    rigsig.eval()
    mnist_text_embed.eval()
    test_loss = 0
    with torch.no_grad():
        for image, label in tqdm(test_dloader, total=len(test_dloader), desc=f"E{e + 1} - TEST"):
            image, label = image.to(device), label.to(device)
            text_cond = mnist_text_embed(label)
            image_coords = generate_coordinates(image)

            pred_splats = rigsig(batch_size, n_splats, text_cond)
            sampled_colors = rigsig.splat_control.sample_col_from_splats(pred_splats, image_coords)
            target_colors = image.view(batch_size, 1, -1).transpose(1, 2)
            loss = nn.functional.mse_loss(sampled_colors, target_colors)
            test_loss += loss.item()

    test_loss /= len(train_dloader)

    print(f"train: {train_loss}, test:{test_loss}")

    with torch.no_grad():
        label = torch.zeros(100, 10).to(device)
        for i in range(10):
            label[i * 10:(i + 1) * 10, i] = 1.0
        text_cond = mnist_text_embed(label)
        splats = rigsig(100, n_splats, text_cond)
        dummy_shape = torch.empty(100, 1, 32, 32, device=device)
        dummy_coords = generate_coordinates(dummy_shape)
        colors = rigsig.splat_control.sample_col_from_splats(splats, dummy_coords)
        sampled_image = colors_to_image(colors, 32, 32)
        render_image(sampled_image)
