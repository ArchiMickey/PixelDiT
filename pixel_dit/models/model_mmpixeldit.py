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

class MMPixelDiT(nn.Module):
    def __init__(
        self,
        *,
        image_size=32,
        semantic_patch_size=16,
        channels=3,
        dit_dim=128,
        pit_dim=64,
        dit_depth=4,
        pit_depth=4,
        heads=4,
        dim_head=32,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=10,
        num_registers=4,
        class_dropout_prob=0.1,
        compress_ratio=2,
        use_abs_pe=True,
        qk_norm=False
    ):
        super().__init__()
        self.image_size = image_size
        self.semantic_patch_size = semantic_patch_size
        self.use_abs_pe = use_abs_pe
        self.compress_ratio = compress_ratio
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.class_dropout_prob = class_dropout_prob
        
        self.base_h_sem = self.base_w_sem = image_size // semantic_patch_size
        self.base_h_pix = self.base_w_pix = image_size
        
        # --- 1. Semantic Stream (MMDiT) ---
        self.to_semantic_patch = nn.Sequential(
            nn.Conv2d(channels, dit_dim, kernel_size=semantic_patch_size, stride=semantic_patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        if use_abs_pe:
            self.semantic_pos_embedding = nn.Parameter(torch.randn(1, self.base_h_sem * self.base_w_sem, dit_dim))
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dit_dim),
            nn.Linear(dit_dim, dit_dim * 4),
            nn.SiLU(),
            nn.Linear(dit_dim * 4, dit_dim)
        )
        
        self.class_registers = nn.Parameter(torch.randn(num_classes + 1, num_registers, dit_dim))
        self.register_pos_embed = nn.Parameter(torch.randn(1, num_registers, dit_dim))
        
        self.dit_blocks = nn.ModuleList([
            MMDiTBlock(dit_dim, heads, dim_head, mlp_ratio, dropout, qk_norm=qk_norm)
            for _ in range(dit_depth)
        ])

        # --- 2. Pixel Stream (PiT) ---
        self.to_pixel_patch = nn.Sequential(
            nn.Conv2d(channels, pit_dim, kernel_size=1, stride=1),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        if use_abs_pe:
            self.pixel_pos_embedding = nn.Parameter(torch.randn(1, self.base_h_pix * self.base_w_pix, pit_dim))
        self.up_conv = nn.Conv2d(dit_dim, dit_dim, kernel_size=3, padding=1)

        self.pit_blocks = nn.ModuleList([
            PixelDiTBlock(pit_dim, heads, dit_dim, dim_head, mlp_ratio, dropout, compress_ratio, qk_norm=qk_norm)
            for _ in range(pit_depth)
        ])

        self.final_layer = PixelDiTFinalLayer(pit_dim, dit_dim, channels)
        
    def forward(self, x, times, y=None):
        b, c, h, w = x.shape
        
        # --- Semantic Processing ---
        s = self.to_semantic_patch(x)
        h_sem, w_sem = h // self.semantic_patch_size, w // self.semantic_patch_size
        
        if self.use_abs_pe:
            pe_sem = interpolate_pos_embed(self.semantic_pos_embedding, self.base_h_sem, self.base_w_sem, h_sem, w_sem)
            s = s + pe_sem
        
        t_emb = self.time_mlp(times)
        if y is None:
            y = torch.full((b,), self.num_classes, device=x.device, dtype=torch.long)
            
        c_regs = self.class_registers[y]
        c_regs = c_regs + self.register_pos_embed
        c_global = c_regs.mean(dim=1)
        condition_emb = t_emb + c_global

        # Compute RoPE for Semantic Path
        freqs_cis_sem = compute_rope_2d(self.dit_blocks[0].attn.dim_head, h_sem, w_sem, device=x.device)

        for block in self.dit_blocks:
            s, c_regs = block(s, c_regs, condition_emb, freqs_cis_sem, h_sem, w_sem)

        # Bridge: Upsample Semantic Outputs to condition Pixel Stream
        s_img = rearrange(s, 'b (h w) c -> b c h w', h=h_sem, w=w_sem)
        s_up = F.interpolate(s_img, (h, w), mode='bilinear', align_corners=False)
        s_up = self.up_conv(s_up)
        cond_pit = rearrange(s_up, 'b c h w -> b (h w) c')
        
        # --- Pixel Processing ---
        p = self.to_pixel_patch(x)
        if self.use_abs_pe:
            pe_pix = interpolate_pos_embed(self.pixel_pos_embedding, self.base_h_pix, self.base_w_pix, h, w)
            p = p + pe_pix

        # Compute RoPE for Pixel Path
        h_comp, w_comp = h // self.compress_ratio, w // self.compress_ratio
        freqs_cis_pix = compute_rope_2d(self.pit_blocks[0].attn.dim_head, h_comp, w_comp, device=x.device)

        for block in self.pit_blocks:
            p = block(p, cond_pit, h, w, freqs_cis_pix)

        x = self.final_layer(p, cond_pit)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

def MMPixelDiT_Tiny(**kwargs):
    kwargs.pop('num_classes', None)
    compress_ratio = kwargs.pop('compress_ratio', 4)
    return MMPixelDiT(
        dit_dim=128, 
        pit_dim=16, 
        dit_depth=6, 
        pit_depth=2, 
        heads=4, 
        dim_head=32, 
        mlp_ratio=4.0, 
        qk_norm='rmsnorm', 
        compress_ratio=compress_ratio, 
        num_registers=4, 
        **kwargs
    )

def MMPixelDiT_Tiny_4(**kwargs):
    return MMPixelDiT_Tiny(semantic_patch_size=4, compress_ratio=4, **kwargs)

def create_mmpixeldit_model(model_name, **kwargs):
    if model_name.lower() == 'mmpixeldit-tiny/4':
        return MMPixelDiT_Tiny_4(**kwargs)
    raise ValueError(f"Unsupported MMPixelDiT model: {model_name}")
