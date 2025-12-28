import torch
from torch import nn

import torch
from torch import nn


class SplatControl(nn.Module):
    def __init__(
            self,
            c_dim: int,
            num_pos_freq: int = 8,
            num_col_freq: int = 5,
    ):
        super().__init__()
        self.c_dim = int(c_dim)
        self.num_pos_freq = int(num_pos_freq)
        self.num_col_freq = int(num_col_freq)

        pos_powers = torch.arange(self.num_pos_freq, device='cuda' if torch.cuda.is_available() else 'cpu') - 3
        self.register_buffer('pos_frequencies', torch.pi * (2.0 ** pos_powers))

        col_powers = torch.arange(self.num_col_freq, device='cuda' if torch.cuda.is_available() else 'cpu') - 3
        self.register_buffer('col_frequencies', torch.pi * (2.0 ** col_powers))

        # Embedding dims: sin/cos for each freq
        pos_embed_dim = 2 * self.num_pos_freq * 3  # 2 (sin/cos) * freqs * 3 (xyz)
        col_embed_dim = 2 * self.num_col_freq * self.c_dim  # Optional: embed colors similarly
        other_dim = 3 + 4 + 1  # scale (3), rotation quat (4), opacity (1)

        self.base_dim = pos_embed_dim + col_embed_dim + other_dim
        self.final_dim = 3 + self.c_dim + 3 + 4 + 1  # pos(3), col(c_dim), scale(3), quat(4), op(1)

    def positional_encoding(self, inputs, frequencies):
        # inputs: [..., d] where d is 3 for pos or c_dim for col
        # frequencies: [num_freq]
        scaled = inputs[..., None] * frequencies  # [..., d, num_freq]
        sins = torch.sin(scaled)
        coss = torch.cos(scaled)
        return torch.cat([sins, coss], dim=-1).flatten(-2)  # [..., 2*num_freq*d]

    def gen_rand_splats(self, batch_size, num_splats):
        device = self.pos_frequencies.device
        b, n = batch_size, num_splats

        # Raw params
        pos = torch.empty(b, n, 3, device=device).uniform_(-1, 1)  # [-1, 1]
        col = torch.empty(b, n, self.c_dim, device=device).uniform_(0, 1)  # [0, 1]
        scale = torch.empty(b, n, 3, device=device).uniform_(0.01, 0.1)  # Small positive scales
        quat = torch.randn(b, n, 4, device=device)
        quat = quat / quat.norm(dim=-1, keepdim=True)  # Unit quaternion
        opacity = torch.empty(b, n, 1, device=device).uniform_(0.5, 1.0)  # [0.5, 1] to avoid too many dead splats

        # Embed pos and col with Fourier (RoPE-like)
        pos_embed = self.positional_encoding(pos, self.pos_frequencies)
        col_embed = self.positional_encoding(col, self.col_frequencies)  # Optional, but since you asked

        # Concat embeds + raw others
        splats = torch.cat([pos_embed, col_embed, scale, quat, opacity], dim=-1)

        return splats

    def cleanup_splats(self, splats):
        assert splats.ndim == 3 and splats.size(-1) == self.final_dim, f"splats wrong size: {splats.shape}"

        b, n, _ = splats.shape

        # Use sequential splitting with fixed sizes
        splits = torch.split(splats, [3, self.c_dim, 3, 4, 1], dim=-1)
        # splits is a tuple of 5 tensors: (pos, col, scale, quat, opacity)
        pos, col, scale, quat, opacity = splits

        # Apply activations
        clean_pos = torch.tanh(pos)
        clean_col = torch.sigmoid(col)
        clean_scale = torch.nn.functional.softplus(scale)
        clean_quat = torch.nn.functional.normalize(quat, dim=-1)
        clean_opacity = torch.sigmoid(opacity)

        return torch.cat([clean_pos, clean_col, clean_scale, clean_quat, clean_opacity], dim=-1)

    def quat_to_rotmat(self, quat):
        # Assumes unit quat [b, n, 4]
        w, x, y, z = quat.unbind(-1)
        wx, wy, wz = 2.0 * w * x, 2.0 * w * y, 2.0 * w * z
        xx, xy, xz = 2.0 * x * x, 2.0 * x * y, 2.0 * x * z
        yy, yz, zz = 2.0 * y * y, 2.0 * y * z, 2.0 * z * z

        r00 = 1.0 - yy - zz
        r01 = xy - wz
        r02 = xz + wy
        r10 = xy + wz
        r11 = 1.0 - xx - zz
        r12 = yz - wx
        r20 = xz - wy
        r21 = yz + wx
        r22 = 1.0 - xx - yy

        return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], -1).view(*quat.shape[:-1], 3, 3)

    def sample_col_from_splats(self, clean_splats, coordinates):
        assert clean_splats.ndim == 3 and clean_splats.size(-1) == self.final_dim
        assert coordinates.ndim == 3 and coordinates.size(-1) == 3

        b, n, _ = clean_splats.shape
        m = coordinates.size(1)

        # Safe split into all parts at once
        mu, col, scale, quat, opacity = torch.split(
            clean_splats, [3, self.c_dim, 3, 4, 1], dim=-1
        )  # each: [b, n, size]

        # Rotation matrix [b,n,3,3]
        R = self.quat_to_rotmat(quat)

        # Covariance Σ = R * diag(scale^2) * R^T
        S = torch.diag_embed(scale ** 2)  # [b,n,3,3]
        Sigma = R @ S @ R.transpose(-2, -1)  # [b,n,3,3]

        # Difference vectors [b,m,n,3]
        diff = coordinates.unsqueeze(2) - mu.unsqueeze(1)  # broadcasting

        # Mahalanobis distance squared: diff^T * Σ^{-1} * diff
        Sigma_inv = torch.inverse(Sigma)  # [b,n,3,3]
        mahal = torch.einsum('bmnd,bnde,bmne->bmn', diff, Sigma_inv, diff)

        # Density
        density = torch.exp(-0.5 * mahal)  # [b,m,n]

        # Weights
        weights = density.unsqueeze(-1) * opacity.unsqueeze(1)  # [b,m,n,1]

        # Weighted colors
        weighted_col = weights * col.unsqueeze(1)  # [b,m,n,c_dim]

        # Accumulate
        num = weighted_col.sum(dim=2)  # [b,m,c_dim]
        den = weights.sum(dim=2) + 1e-6  # [b,m,1]

        colors = num / den  # [b,m,c_dim]

        return colors


# class SplatControl(nn.Module):
#     def __init__(
#             self,
#             c_dim: int,
#             num_pos_freq: int = 8,
#             num_col_freq: int = 5,
#     ):
#         super().__init__()
#         self.c_dim = int(c_dim)
#         self.num_pos_freq = int(num_pos_freq)
#         self.num_col_freq = int(num_col_freq)
#
#         pos_powers = torch.arange(self.num_pos_freq) - 3
#         pos_frequencies = torch.pi * (2.0 ** pos_powers)
#         col_powers = torch.arange(self.num_col_freq) - 3
#         col_frequencies = torch.pi * (2.0 ** col_powers)
#
#         self.base_dim = int(pos_frequencies * 3 + col_frequencies * self.c_dim + 3 + 4 + 1)
#         self.final_dim = int(3 + self.c_dim + 3 + 4 + 1)
#
#     def gen_rand_splats(self, batch_size, num_splats):
#         b, n = batch_size, num_splats
#         pos = torch.empty(b, n, 3).uniform_(-1, 1)  # 3, must be [-1, 1]
#         col = torch.empty(b, n, self.c_dim).uniform_(0, 1)  # c_dim, must be [0, 1]
#         scale = torch.randn(b, n, 3).abs_()  # 3, must be [0, inf)
#         rotation =  # 4, must be
#         opacity = torch.empty(b, n, 1).uniform_(0, 1)  # 1, must be [0, 1]
#
#         # TODO apply RoPE on pos and color
#         # TODO figure out wtf the values in rotation quaternion must be, assume to be sin/cos of angle?
#
#         splats = torch.cat([pos, col, scale, rotation, opacity], dim=-1)
#
#         return splats
#
#     def cleanup_splats(self, splats):
#         assert splats.ndim == 3, "splats must be [b, l, e] size"
#         assert splats.size(-1) == self.final_dim, "splats wrong size in cleanup"
#         # TODO implement logic to turn "raw" splats into actually usable ones with the activation functions and stuff
#         return clean_splats
#
#     def sample_col_from_splats(self, clean_splats, coordinates):
#         pass


class Transformer(nn.Module):
    def __init__(self, d_dim: int, num_heads: int, base_dropout: float, cross_dropout: float, mlp_dropout: float):
        super().__init__()

        self.base_norm = nn.LayerNorm(d_dim)
        self.base_mha = nn.MultiheadAttention(
            embed_dim=d_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=base_dropout,
        )
        self.base_residual = nn.Parameter(torch.randn(d_dim) * 1e-3)

        self.cross_norm = nn.LayerNorm(d_dim)
        self.cross_mha = nn.MultiheadAttention(
            embed_dim=d_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=cross_dropout,
        )
        self.cross_residual = nn.Parameter(torch.randn(d_dim) * 1e-3)

        self.mlp_norm = nn.LayerNorm(d_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_dim, d_dim * 4),
            nn.SiLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(d_dim * 4, d_dim)
        )
        self.mlp_residual = nn.Parameter(torch.randn(d_dim) * 1e-3)

        self.skip_scalar = nn.Parameter(torch.randn(d_dim) * 1e-3)

    def forward(self, splats, text_tokens):
        w_splats = splats

        base_norm = self.base_norm(w_splats)
        base_attn_out, _ = self.base_mha(base_norm, base_norm, base_norm, need_weights=False)
        w_splats = w_splats + base_attn_out * self.base_residual

        cross_norm = self.cross_norm(w_splats)
        cross_attn_out, _ = self.cross_mha(cross_norm, text_tokens, text_tokens, need_weights=False)
        w_splats = w_splats + cross_attn_out * self.cross_residual

        mlp_norm = self.mlp_norm(w_splats)
        mlp_out = self.mlp(mlp_norm)
        w_splats = w_splats + mlp_out * self.mlp_residual

        splats = splats + w_splats * self.skip_scalar
        return splats


class RIGSIG(nn.Module):
    def __init__(
            self,
            num_pos_freq: int,
            num_col_freq: int,
            c_dim: int,
            d_dim: int,
            num_blocks: int,
            num_heads: int,
            base_attn_dropout: float,
            cross_attn_dropout: float,
            mlp_dropout: float,
    ):
        super().__init__()
        self.num_pos_freq = int(num_pos_freq)
        self.num_col_freq = int(num_col_freq)
        self.c_dim = int(c_dim)
        self.d_dim = int(d_dim)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)

        self.splat_control = SplatControl(
            c_dim=c_dim,
            num_pos_freq=num_pos_freq,
            num_col_freq=num_col_freq,
        )

        self.base_dim = self.splat_control.base_dim
        self.final_dim = self.splat_control.final_dim

        self.splat_encode = nn.Linear(self.base_dim, d_dim)

        self.blocks = nn.ModuleList([
            Transformer(
                d_dim=d_dim,
                num_heads=num_heads,
                base_dropout=base_attn_dropout,
                cross_dropout=cross_attn_dropout,
                mlp_dropout=mlp_dropout,
            ) for _ in range(num_blocks)
        ])

        self.splat_decode = nn.Linear(d_dim, self.final_dim)

    def forward(self, batch_size, num_splats, text_tokens):
        assert text_tokens.ndim == 3, "tokens must be [b, l, e] size"
        assert text_tokens.size(-1) == self.d_dim, "text tokens must fit d_dim"

        raw_splats = self.splat_control.gen_rand_splats(batch_size, num_splats)
        latent_splats = self.splat_encode(raw_splats)

        for block in self.blocks:
            latent_splats = block(latent_splats, text_tokens)

        pred_splats = self.splat_decode(latent_splats)

        clean_splats = self.splat_control.cleanup_splats(pred_splats)

        return clean_splats
