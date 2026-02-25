import torch
import torch.nn as nn
import torch.nn.functional as F

def timeshift(t, s=1.0):
    return t/(t+(1-t)*s)

class RectifiedFlow(nn.Module):
    def __init__(self, model, class_dropout_prob=0.1, t_eps=5e-2, P_mean=-0.8, P_std=0.8):
        super().__init__()
        self.model = model
        self.class_dropout_prob = class_dropout_prob
        self.t_eps = float(t_eps)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)

    def sample_t(self, n, device):
        t = torch.rand((n,), device=device) * self.P_std + self.P_mean
        return torch.sigmoid(t)

    def forward(self, x_1, y=None):
        b, c, h, w = x_1.shape
        device = x_1.device

        t = self.sample_t(b, device)
        x_0 = torch.randn_like(x_1)
        
        t_view = t.view(b, 1, 1, 1)
        x_t = t_view * x_1 + (1 - t_view) * x_0

        if y is not None and self.class_dropout_prob > 0:
            probs = torch.full(y.shape, self.class_dropout_prob, device=y.device)
            mask = torch.bernoulli(probs).to(torch.bool)
            y = torch.where(mask, torch.full_like(y, self.model.num_classes), y)

        # Model predicts x_1 directly
        pred_x1 = self.model(x_t, t, y)

        # Derive velocity: v = (x_1 - x_t) / (1 - t)
        v_pred = (pred_x1 - x_t) / (1 - t_view).clamp(min=self.t_eps)
        v_target = (x_1 - x_t) / (1 - t_view).clamp(min=self.t_eps) 

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, shape, steps=50, device='cpu', y=None, cfg_scale=1.0, shift=1.0, cfg_interval=[0.1, 1.0], return_all=False):
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps
        samples = [x]

        for i in range(steps):
            t_val = i / steps
            t_shifted = timeshift(t_val, shift)
            t = torch.full((shape[0],), t_shifted, device=device)
            
            # Internal helper to calculate velocity from the x1 prediction
            def get_v(x_curr, t_curr, y_curr):
                pred_x1 = self.model(x_curr, t_curr, y_curr)
                return (pred_x1 - x_curr) / max(1.0 - t_shifted, self.t_eps)

            use_cfg = cfg_scale > 1.0 and y is not None and (cfg_interval[0] <= t_val <= cfg_interval[1])

            if use_cfg:
                # CFG Logic: Calculate conditioned and unconditioned velocities
                y_null = torch.full_like(y, self.model.num_classes)
                
                # Batched for speed
                x_twice = torch.cat([x, x], dim=0)
                t_twice = torch.cat([t, t], dim=0)
                y_twice = torch.cat([y, y_null], dim=0)
                
                # We need to compute pred_x1 for both, then derive v
                pred_x1_both = self.model(x_twice, t_twice, y_twice)
                v_both = (pred_x1_both - x_twice) / max(1.0 - t_shifted, self.t_eps)
                
                v_cond, v_uncond = v_both.chunk(2, dim=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = get_v(x, t, y)
                
            x = x + v * dt
            
            if return_all:
                samples.append(x)

        if return_all:
            return torch.stack(samples)
        return x