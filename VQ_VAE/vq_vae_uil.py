def compute_loss(x, recon_x, z_e, z_q):
    recon_loss = F.mse_loss(recon_x, x)
    vq_loss = F.mse_loss(z_q.detach(), z_e)  # Move codebook towards encoder
    commit_loss = F.mse_loss(z_e, z_q.detach())  # Commit encoder to codebook
    
    return recon_loss + vq_loss + 0.25 * commit_loss
    
def evaluate_VQVAE(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_loss = 0 
    n_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            z_e , z_q, recon_x  = model(x)
            
            loss = compute_loss(x,recon_x, z_e, z_q)
            
            total_loss += loss.item()
            
            n_batches += 1 
            
    return total_loss/ n_batches