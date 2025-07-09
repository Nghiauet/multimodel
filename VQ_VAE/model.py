class residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        
    def forward(self, x):
        return x + self.residual_block(x)
    
class VQVAE(nn.Module):
    def __init__(self, dim,K,D ):
        super().__init__()
        assert dim == D, f"Encoder output dim ({dim}) must match codebook dim ({D})"
        self.encoder_residual_1= residual(dim)
        self.encoder_residual_2= residual(dim)
        
        self.decoder_residual_1= residual(dim)
        self.decoder_residual_2= residual(dim)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),  # 8*8
            self.encoder_residual_1,
            self.encoder_residual_2,
        )
        
        self.decoder = nn.Sequential(
            self.decoder_residual_1,
            self.decoder_residual_2,
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(dim,dim,4,2,1), # 16*16
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(dim,3,4,2,1), # 32*32
        )
        
        self.codebook = nn.Embedding(K,D)
        nn.init.uniform_(self.codebook.weight, -1/K, 1/K)
    def encode(self,x):
        z_e = self.encoder(x)
        batch_size, channels, height, width = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_e_flat = z_e_flat.view(-1, channels)  # [B*H*W, C]
        
        
        distances = torch.sum((z_e_flat.unsqueeze(1) - self.codebook.weight.unsqueeze(0))**2, dim=-1) 
        # z_e_flat -> [B*H*W, C] -> [B*H*W,1,  C]
        # self.codebook.weight.unsqueeze(0) -> [K, D] -> [1, K, D]
        # distances -> [B*H*W, K]
        

        k = torch.argmin(distances, dim=-1)  # [B*H*W]
        # k [B*H*W] -> [B*H*W, 1]
        z_q_flat = self.codebook(k)  # [B*H*W, D]
        z_q = z_q_flat.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()  # Back to [B, C, H, W]
        
        z_q = z_e + (z_q - z_e).detach() # Straight-through estimator
        return z_e, z_q
        
    def decode(self, z_q):
        x_recon = self.decoder(z_q)
        return x_recon
    
    def forward(self, x):
        z_e, z_q = self.encode(x)
        
        x_recon = self.decode(z_q)
        
        return z_e, z_q , x_recon
    # Add this method to the VQVAE class to extract discrete tokens
    def get_tokens(self, x):
        """Extract discrete tokens from input images"""
        z_e = self.encoder(x)
        batch_size, channels, height, width = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_e_flat = z_e_flat.view(-1, channels)  # [B*H*W, C]
        
        distances = torch.sum((z_e_flat.unsqueeze(1) - self.codebook.weight.unsqueeze(0))**2, dim=-1)
        tokens = torch.argmin(distances, dim=-1)  # [B*H*W]
        
        return tokens.view(batch_size, height, width)  # [B, H, W]

    def decode_tokens(self, tokens):
        """Decode discrete tokens back to images"""
        batch_size, height, width = tokens.shape
        z_q_flat = self.codebook(tokens.view(-1))  # [B*H*W, D]
        z_q = z_q_flat.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)

