import torch
import torch.nn.functional as F
from .base_trainer import BaseTrainer


class VQVAETrainer(BaseTrainer):
    """VQ-VAE specific trainer"""
    
    def __init__(self, model, optimizer, device, gradient_clip_norm=1.0):
        super().__init__(model, optimizer, device, gradient_clip_norm)
    
    def _compute_loss(self, batch):
        """Compute VQ-VAE loss"""
        x = batch[0]  # Images are first element in batch
        
        # Forward pass
        z_e, z_q, recon_x = self.model(x)
        
        # Compute losses
        recon_loss = F.mse_loss(recon_x, x)
        vq_loss = F.mse_loss(z_q.detach(), z_e)  # Move codebook towards encoder
        commit_loss = F.mse_loss(z_e, z_q.detach())  # Commit encoder to codebook
        
        total_loss = recon_loss + vq_loss + 0.25 * commit_loss
        return total_loss