import sys
import os
import torch
import torch.nn as nn
from vit_pytorch import ViT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae import VQVAE
from models.encoder import Encoder

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))
    
class ConvPrior(nn.Module):
    def __init__(self, cls_dim=1024, n_embed=128, small_size=16, out_size=32):
        super().__init__()
        self.small_size = small_size
        self.out_size = out_size
        self.n_embed = n_embed
        hidden_dim = n_embed * small_size * small_size

        # 1) CLS → 16x16 latent map
        self.fc = nn.Sequential(
            nn.Linear(cls_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 2) Upsample: 16x16 → 32x32
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(n_embed, n_embed, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # 3) Residual block(s) at 32x32
        self.res_block = ResidualBlock(n_embed)
        
        # 4) Final projection to match n_embeddings
        self.final_proj = nn.Conv2d(n_embed, n_embed, kernel_size=1)

    def forward(self, x):
        # x: [B, cls_dim]
        h = self.fc(x)  # [B, C*16*16]
        h = h.view(-1, self.n_embed, self.small_size, self.small_size)  # [B, C, 16, 16]
        h = self.upsample(h)  # [B, C, 32, 32]
        h = self.res_block(h)  # [B, C, 32, 32]
        logits = self.final_proj(h)  # [B, C, 32, 32]
        return logits

class VITMLP(nn.Module):
    """
    Combined model that uses ViT features to condition VQVAE latent space predictions.
    Architecture:
    1. Pretrained ViT (frozen) extracts [CLS] token features
    2. Lightweight MLP predicts P(z_i | f_ViT) over VQVAE codebook indices
    3. Pretrained VQVAE (frozen) provides latent space structure
    """
    def __init__(self, args):
        super().__init__()
        print("\n=== vit_mlp.py - Initialization ===")
        
        self.device = torch.device(args.device if args.device else 'cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize ViT architecture
        self.vit = ViT(
            image_size=256,
            patch_size=32,
            num_classes=2,  # binary classification
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            channels=3,
            dropout=0.1,
            emb_dropout=0.1
        ).to(self.device)
        
        # Load ViT checkpoint if provided
        if args.vit_checkpoint != 'dummy.pth':
            vit_state = torch.load(args.vit_checkpoint, map_location=self.device)
            if 'model_state' in vit_state:
                state_dict = vit_state['model_state']
            elif 'state_dict' in vit_state:
                state_dict = vit_state['state_dict']
            else:
                state_dict = vit_state
            
            # Load state dict directly since the checkpoint already has the correct structure
            self.vit.load_state_dict(state_dict)
            
        print("ViT model initialized")
        print("==============================\n")
        
        # Freeze ViT weights
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Initialize VQVAE
        self.vqvae = VQVAE(
            h_dim=256,
            res_h_dim=64,
            n_res_layers=4,
            n_embeddings=args.n_embeddings,
            embedding_dim=64,
            beta=0.5,
            dropout=0.2
        ).to(self.device)
        
        # Load VQVAE checkpoint if provided
        if args.vqvae_checkpoint != 'dummy.pth':
            vqvae_state = torch.load(args.vqvae_checkpoint, map_location=self.device)
            if 'state_dict' in vqvae_state:
                state_dict = vqvae_state['state_dict']
            elif 'model_state' in vqvae_state:
                state_dict = vqvae_state['model_state']
            else:
                state_dict = vqvae_state
            
            # Remove 'model.' prefix from all keys
            vqvae_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            self.vqvae.load_state_dict(vqvae_state_dict)
        
        # Freeze VQVAE weights
        for param in self.vqvae.parameters():
            param.requires_grad = False
            
        # Initialize PriorDeconv with VQVAE's n_embeddings
        self.mlp_prior = ConvPrior(
            cls_dim=1024,
            n_embed=args.n_embeddings,
            out_size = 32,
            small_size = 16
        ).to(self.device)
        
        # Load MLP checkpoint if provided
        if hasattr(args, 'mlp_checkpoint') and args.mlp_checkpoint:
            mlp_state = torch.load(args.mlp_checkpoint, map_location=self.device)
            if 'model_state' in mlp_state:
                state_dict = mlp_state['model_state']
            elif 'state_dict' in mlp_state:
                state_dict = mlp_state['state_dict']
            else:
                state_dict = mlp_state
            
            # Filter for MLP weights only
            mlp_state_dict = {k.replace('mlp_prior.', ''): v 
                            for k, v in state_dict.items() 
                            if k.startswith('mlp_prior')}
            
            if mlp_state_dict:
                self.mlp_prior.load_state_dict(mlp_state_dict)
        
    def forward(self, x, return_probs=False):
        print("\n=== vit_mlp.py - Forward Method ===")
        print("Initial input shape:", x.shape)
        print("Initial input type:", x.dtype)
        print("Initial input device:", x.device)
        
        # Resize input for ViT and convert to 3 channels
        if x.shape[-1] != 256:  # If input is not 256x256
            x = torch.nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        if x.shape[1] == 1:  # If input is grayscale
            x = x.repeat(1, 3, 1, 1)  # Convert to RGB by repeating the channel
        print("After preprocessing shape:", x.shape)
            
        # Extract ViT features
        print("Before to_patch_embedding shape:", x.shape)
        x = self.vit.to_patch_embedding(x)
        print("After to_patch_embedding shape:", x.shape)
        b, n, _ = x.shape  # batch, num_patches, dim
        print("==============================\n")
        
        # Add positional embeddings and apply dropout
        x += self.vit.pos_embedding[:, :n]
        x = self.vit.dropout(x)
        
        # Get [CLS] token from transformer
        cls_tokens = self.vit.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.vit.transformer(x)
        
        # Extract [CLS] token (first token)
        cls_token = x[:, 0]  # shape: [batch, dim]
        
        # Predict codebook logits
        logits = self.mlp_prior(cls_token)  # shape: [batch, n_embed, 32, 32]
        
        if return_probs:
            # Convert to probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=1)
            return logits, probs
        return logits

    def compute_nll(self, x):
        """Compute negative log-likelihood for anomaly detection"""
        # Get predicted probabilities
        _, probs = self.forward(x, return_probs=True)
        
        # Get actual VQVAE encoding indices
        encoding_indices = self.get_vqvae_latents(x)
        
        # Gather probabilities for actual indices
        selected_probs = torch.gather(probs, 1, encoding_indices.unsqueeze(1)).squeeze(1)
        
        # Compute negative log likelihood
        nll = -torch.log(selected_probs + 1e-10)  # Add small epsilon for numerical stability
        nll = nll.mean(dim=(1,2))  # Average over spatial dimensions
        
        return nll

    def predict_anomaly(self, x, threshold=0.5):
        """Predict anomaly scores and binary labels"""
        nll = self.compute_nll(x)
        # Normalize NLL using min-max scaling based on expected range
        min_nll, max_nll = 0.0, 10.0  # Expected NLL range
        scores = (nll - min_nll) / (max_nll - min_nll)
        scores = torch.clamp(scores, 0.0, 1.0)  # Ensure [0,1] range
        labels = (scores > threshold).float()
        return scores, labels
        
    def reconstruct(self, x):
        """Generate reconstruction from input image"""
        with torch.no_grad():
            # Get predicted latent distribution
            logits = self.forward(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Sample from predicted distribution
            encoding_indices = torch.argmax(probs, dim=1)
            
            # Flatten indices to match VQVAE expected format
            encoding_indices = encoding_indices.view(-1)
            
            # Get quantized vectors directly from codebook
            quantized = self.vqvae.vector_quantization.embedding(encoding_indices)
            
            # Get embedding dimension from codebook
            embedding_dim = self.vqvae.vector_quantization.embedding.weight.shape[1]
            
            # Reshape to [B, C, H, W] expected by decoder
            # 디코더의 입력 채널 수에 맞게 조정
            quantized = quantized.view(
                x.size(0), 
                embedding_dim,
                logits.size(2), 
                logits.size(3)
            )
            
            # 디코더의 입력 채널 수 확인 및 조정
            if quantized.size(1) != self.vqvae.decoder.inverse_conv_stack[0].in_channels:
                # 필요한 경우 채널 수 조정
                quantized = torch.nn.functional.interpolate(
                    quantized,
                    size=(quantized.size(2), quantized.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Decode to reconstruction
            reconstruction = self.vqvae.decoder(quantized)
            
            # 출력 크기가 입력과 일치하는지 확인
            if reconstruction.size() != x.size():
                reconstruction = torch.nn.functional.interpolate(
                    reconstruction,
                    size=(x.size(2), x.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                
        return reconstruction
        
    def get_anomaly_map(self, x):
        """Generate anomaly localization map in image space"""
        with torch.no_grad():
            # 1. Get reconstruction
            reconstruction = self.reconstruct(x)
            
            # 2. Calculate pixel-wise difference
            if x.shape[1] == 1:  # If input is grayscale
                diff = torch.abs(x - reconstruction)
            else:  # If input is RGB
                diff = torch.abs(x.mean(dim=1, keepdim=True) - 
                               reconstruction.mean(dim=1, keepdim=True))
            
            # 3. Get NLL from prior
            _, probs = self.forward(x, return_probs=True)
            encoding_indices = self.get_vqvae_latents(x)
            selected_probs = torch.gather(probs, 1, encoding_indices.unsqueeze(1)).squeeze(1)
            nll = -torch.log(selected_probs + 1e-10)
            
            # 4. Combine reconstruction error with NLL
            # Resize NLL to match image size
            nll = torch.nn.functional.interpolate(
                nll.unsqueeze(1),
                size=(x.shape[2], x.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            
            # Combine both signals
            anomaly_map = (diff + nll) / 2.0
            
            return anomaly_map.squeeze(1)

    def get_vqvae_latents(self, x):
        """Helper to get VQVAE encoding indices for training, downsampled to out_size x out_size"""
        with torch.no_grad():
            # Convert to grayscale if input is RGB
            if x.shape[1] == 3:
                x = x.mean(dim=1, keepdim=True)  # Convert to grayscale by averaging channels
            
            _, _, _, _, encoding_indices, _, _ = self.vqvae(x, return_extra=True)
            
            # Downsample encoding indices to 32x32
            encoding_indices = encoding_indices.float()  # Convert to float for interpolation
            encoding_indices = torch.nn.functional.interpolate(
                encoding_indices.unsqueeze(1),  # Add channel dimension [B, 1, H, W]
                size=(32, 32),
                mode='nearest'  # Use nearest neighbor to preserve discrete indices
            )
            encoding_indices = encoding_indices.squeeze(1).long()  # Remove channel dimension and convert back to long
            
            return encoding_indices

