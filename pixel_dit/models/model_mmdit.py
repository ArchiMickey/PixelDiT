import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from einops import rearrange
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# --- Utilities ---

def interpolate_pos_embed(pos_embed, old_h, old_w, new_h, new_w):
    """Interpolates 1D absolute positional embeddings to a new 2D grid size."""
    if old_h == new_h and old_w == new_w:
        return pos_embed
    dim = pos_embed.shape[-1]
    # Reshape to spatial grid, interpolate, and flatten back
    pos_embed = pos_embed.reshape(1, old_h, old_w, dim).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(pos_embed, size=(new_h, new_w), mode='bicubic', align_corners=False)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
    return pos_embed

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def modulate(x, scale, shift):
    return x * scale + shift

# --- RoPE ---

def apply_rope(q, k, freqs_cis):
    """Applies computed RoPE frequencies to queries and keys."""
    # q, k: (B, H, N, D)
    # freqs_cis: (N, D/2)
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    
    # freqs_cis is (N, D/2)
    # q_ is (B, H, N, D/2)
    # We need to align N dimension
    # q_ shape: (B, H, N, D/2)
    # freqs_cis shape: (N, D/2)
    freqs_cis = freqs_cis[None, None, :, :] # (1, 1, N, D/2)
    
    # q_ shape: (B, H, N, D/2)
    # freqs_cis shape: (1, 1, N, D_rope/2)
    
    d_rope_half = freqs_cis.shape[-1]
    d_q_half = q_.shape[-1]
    
    if d_rope_half < d_q_half:
        # Apply RoPE only to the first d_rope_half dimensions
        q_rope = q_[..., :d_rope_half] * freqs_cis
        q_pass = q_[..., d_rope_half:]
        q_out = torch.cat([torch.view_as_real(q_rope).flatten(3), torch.view_as_real(q_pass).flatten(3)], dim=-1)
        
        k_rope = k_[..., :d_rope_half] * freqs_cis
        k_pass = k_[..., d_rope_half:]
        k_out = torch.cat([torch.view_as_real(k_rope).flatten(3), torch.view_as_real(k_pass).flatten(3)], dim=-1)
    else:
        # Truncate freqs_cis if it's larger (shouldn't happen with our setup but for safety)
        freqs_cis = freqs_cis[..., :d_q_half]
        q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
        k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
        
    return q_out.type_as(q), k_out.type_as(k)

def compute_rope_2d(dim_head, h, w, theta=10000.0, device=None):
    """Computes 2D Rotary Positional Embeddings (Legacy for DiT/PixelDiT)."""
    dim_half = dim_head // 2
    
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim_half, 2, device=device).float() / dim_half))
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim_half, 2, device=device).float() / dim_half))
    
    y = torch.arange(h, device=device)
    x = torch.arange(w, device=device)
    
    freqs_y = torch.outer(y, freqs_y)
    freqs_x = torch.outer(x, freqs_x)
    
    freqs_y = freqs_y.unsqueeze(1).expand(h, w, -1)
    freqs_x = freqs_x.unsqueeze(0).expand(h, w, -1)
    
    freqs = torch.cat([freqs_y, freqs_x], dim=-1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).view(-1, freqs.shape[-1])
    return freqs_cis

def apply_rope_2d(q, k, freqs_cis):
    """Applies computed RoPE frequencies to queries and keys (Legacy for DiT/PixelDiT)."""
    return apply_rope(q, k, freqs_cis)

# --- Compress and Expand ---

class LinearCompress(nn.Module):
    def __init__(self, dim, compress_ratio=2):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.proj = nn.Conv2d(dim, dim * (compress_ratio ** 2), kernel_size=compress_ratio, stride=compress_ratio)
        
    def forward(self, x, h, w):
        # x: (b, L, c)
        # h, w are the spatial dimensions of x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class LinearExpand(nn.Module):
    def __init__(self, dim, compress_ratio=2):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.proj = nn.ConvTranspose2d(dim * (compress_ratio ** 2), dim, kernel_size=compress_ratio, stride=compress_ratio)
        
    def forward(self, x, h_comp, w_comp):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h_comp, w=w_comp)
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

# --- Attention ---

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.1, compress_ratio=1, qk_norm=None):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.compress_ratio = compress_ratio

        self.compress = LinearCompress(dim, compress_ratio) if compress_ratio > 1 else nn.Identity()
        
        inner_dim = self.heads * dim_head
        self.qkv = nn.Linear(dim * (compress_ratio ** 2), inner_dim * 3, bias=False)

        if qk_norm == 'rmsnorm':
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)
        elif qk_norm == 'layernorm':
            self.q_norm = nn.LayerNorm(dim_head)
            self.k_norm = nn.LayerNorm(dim_head)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.proj = nn.Linear(inner_dim, dim * (compress_ratio ** 2))
        self.attn_drop = nn.Dropout(dropout)
        
        self.expand = LinearExpand(dim, compress_ratio) if compress_ratio > 1 else nn.Identity()

    def forward(self, x, h, w, freqs_cis):
        if self.compress_ratio > 1:
            # x is (B, L, C) where L = h * w
            x = self.compress(x, h, w)
            h, w = h // self.compress_ratio, w // self.compress_ratio

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rope(q, k, freqs_cis)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale
        )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)

        if self.compress_ratio > 1:
            out = self.expand(out, h, w)

        return out

class JointAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.1, qk_norm=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        if qk_norm == 'rmsnorm':
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)
        elif qk_norm == 'layernorm':
            self.q_norm = nn.LayerNorm(dim_head)
            self.k_norm = nn.LayerNorm(dim_head)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.proj = nn.Linear(inner_dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, c, freqs_cis):
        # x: (B, N_x, D), c: (B, N_c, D)
        n_x = x.shape[1]
        
        combined = torch.cat([x, c], dim=1) # (B, N_x + N_c, D)
        qkv = self.qkv(combined).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE
        # freqs_cis is (N_img, D/2)
        # q, k are (B, H, N_img + N_reg, D)
        q_img, q_reg = q[:, :, :n_x], q[:, :, n_x:]
        k_img, k_reg = k[:, :, :n_x], k[:, :, n_x:]

        q_img, k_img = apply_rope(q_img, k_img, freqs_cis)

        q = torch.cat([q_img, q_reg], dim=2)
        k = torch.cat([k_img, k_reg], dim=2)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        
        out_x, out_c = out[:, :n_x], out[:, n_x:]
        return out_x, out_c

# --- Blocks ---

class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio=4.0, dropout=0.1, qk_norm=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout=dropout, qk_norm=qk_norm)
        
        self.norm2 = RMSNorm(dim)
        inner_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t_emb, freqs_cis, h, w):
        mod = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [m[:, None, :] for m in mod]

        res = x
        x_norm = modulate(self.norm1(x), scale_msa, shift_msa)
        
        # Self-Attention with RoPE
        out_attn = self.attn(x_norm, h, w, freqs_cis)
        
        x = res + gate_msa * out_attn

        # MLP
        res = x
        x = modulate(self.norm2(x), scale_mlp, shift_mlp)
        x = self.mlp(x)
        x = res + gate_mlp * x
        
        return x

class PixelDiTBlock(nn.Module):
    def __init__(self, pit_dim, heads, dit_dim, dim_head, mlp_ratio=4.0, dropout=0.1, compress_ratio=2, qk_norm=False):
        super().__init__()
        self.compress_ratio = compress_ratio
        
        # --- Path 1: Attention ---
        self.norm1 = RMSNorm(pit_dim)
        self.attn = Attention(pit_dim, heads, dim_head, dropout=dropout, compress_ratio=compress_ratio, qk_norm=qk_norm)
        
        # --- Path 2: FFN ---
        self.norm2 = RMSNorm(pit_dim)
        inner_dim = int(pit_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(pit_dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, pit_dim),
            nn.Dropout(dropout)
        )
        
        # Conditioning MLP maps from dit_dim -> pit_dim parameters
        self.mlp_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dit_dim, 6 * pit_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.mlp_cond[-1].weight)
        nn.init.zeros_(self.mlp_cond[-1].bias)

    def forward(self, x, cond, h, w, freqs_cis):
        mod = self.mlp_cond(cond).chunk(6, dim=-1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = [m[:, None, :] if m.dim() == 2 else m for m in mod]
        
        # --- MHSA Path ---
        res1 = x
        x_attn = modulate(self.norm1(x), gamma1, beta1)
        
        out_attn = self.attn(x_attn, h, w, freqs_cis)
            
        x = res1 + out_attn * alpha1
        
        # --- FFN Path ---
        res2 = x
        x_ffn = modulate(self.norm2(x), gamma2, beta2)
        out_ffn = self.ffn(x_ffn)
        x = res2 + out_ffn * alpha2
        
        return x

class MMDiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio=4.0, dropout=0.1, qk_norm=False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # Shared Norms
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Shared Modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

        # Joint Attention
        self.attn = JointAttention(dim, heads, dim_head, dropout=dropout, qk_norm=qk_norm)

        # Shared MLP
        inner_mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, t_emb, freqs_cis, h, w):
        # x: (B, N_x, D), c: (B, N_c, D), t_emb: (B, D)
        n_x = x.shape[1]
        
        # Shared Modulation
        mod = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [m[:, None, :] for m in mod]

        # Norm and modulate (Shared, optimized by concatenating)
        combined = torch.cat([x, c], dim=1)
        combined_norm = modulate(self.norm1(combined), scale_msa, shift_msa)
        x_norm, c_norm = combined_norm[:, :n_x], combined_norm[:, n_x:]
        
        # Joint Attention
        out_x, out_c = self.attn(x_norm, c_norm, freqs_cis)
        
        # Residual and MLP (Shared, optimized by concatenating)
        x = x + gate_msa * out_x
        c = c + gate_msa * out_c
        
        combined = torch.cat([x, c], dim=1)
        combined_mlp = self.mlp(modulate(self.norm2(combined), scale_mlp, shift_mlp))
        
        x = x + gate_mlp * combined_mlp[:, :n_x]
        c = c + gate_mlp * combined_mlp[:, n_x:]
        
        return x, c

class FinalLayer(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.linear(x)
        return x

class PixelDiTFinalLayer(nn.Module):
    def __init__(self, pit_dim, dit_dim, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(pit_dim)
        self.linear = nn.Linear(pit_dim, out_channels)
        self.mlp_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dit_dim, 2 * pit_dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.mlp_cond[-1].weight, 0)
        nn.init.constant_(self.mlp_cond[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x, c):
        shift, scale = self.mlp_cond(c).chunk(2, dim=-1)
        if shift.dim() == 2:
            shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.linear(x)
        return x

# --- Models ---

class MMDiT(nn.Module):
    def __init__(
        self,
        *,
        image_size=32,
        patch_size=4,
        channels=3,
        dim=128,
        depth=6,
        heads=4,
        dim_head=32,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=10,
        num_registers=4,
        class_dropout_prob=0.1,
        use_abs_pe=True,
        qk_norm=False
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_abs_pe = use_abs_pe
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.class_dropout_prob = class_dropout_prob
        
        self.base_h = self.base_w = image_size // patch_size
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        if use_abs_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.base_h * self.base_w, dim))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

        # Learnable registers for each class
        self.class_registers = nn.Parameter(torch.randn(num_classes + 1, num_registers, dim))
        self.register_pos_embed = nn.Parameter(torch.randn(1, num_registers, dim))

        self.blocks = nn.ModuleList([
            MMDiTBlock(dim, heads, dim_head, mlp_ratio, dropout, qk_norm=qk_norm)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(dim, channels * patch_size ** 2)

    def forward(self, x, times, y=None):
        b, c, h, w = x.shape
        h_patch, w_patch = h // self.patch_size, w // self.patch_size
        
        x = self.to_patch_embedding(x)

        if self.use_abs_pe:
            pe = interpolate_pos_embed(self.pos_embedding, self.base_h, self.base_w, h_patch, w_patch)
            x = x + pe

        t_emb = self.time_mlp(times)
        if y is None:
            y = torch.full((b,), self.num_classes, device=x.device, dtype=torch.long)

        # Get registers for the classes
        c_regs = self.class_registers[y] # (B, num_registers, dim)
        c_regs = c_regs + self.register_pos_embed
        
        # Global conditioning is the average of registers
        c_global = c_regs.mean(dim=1) # (B, dim)
        condition_emb = t_emb + c_global

        # Compute RoPE for Image tokens
        freqs_cis = compute_rope_2d(self.blocks[0].attn.dim_head, h_patch, w_patch, device=x.device)

        for block in self.blocks:
            x, c_regs = block(x, c_regs, condition_emb, freqs_cis, h_patch, w_patch)

        x = self.final_layer(x, condition_emb)

        p = self.patch_size
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p, h=h_patch, w=w_patch)

        return x

def MMDiT_Tiny(**kwargs):
    kwargs.pop('num_classes', None)
    patch_size = kwargs.pop('patch_size', 4)
    return MMDiT(
        dim=128, 
        depth=6, 
        heads=4, 
        dim_head=32, 
        mlp_ratio=4.0, 
        num_classes=10, 
        num_registers=4, 
        patch_size=patch_size,
        **kwargs
    )

def MMDiT_Tiny_4(**kwargs):
    return MMDiT_Tiny(patch_size=4, **kwargs)

def create_mmdit_model(model_name, **kwargs):
    if model_name.lower() == 'mmdit-tiny/4':
        return MMDiT_Tiny_4(**kwargs)
    raise ValueError(f"Unsupported MMDiT model: {model_name}")