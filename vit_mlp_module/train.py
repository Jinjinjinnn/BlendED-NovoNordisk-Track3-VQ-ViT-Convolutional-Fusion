import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from vit_mlp import VITMLP
import numpy as np
import os
from tqdm import tqdm
import argparse
from datasets.dataset import get_vitmlp_dataloaders
import wandb
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_checkpoint', type=str, required=True, help='Path to ViT checkpoint')
    parser.add_argument('--vqvae_checkpoint', type=str, required=True, help='Path to VQVAE checkpoint')
    parser.add_argument('--save_dir', type=str, default='vaexvit_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--n_embeddings', type=int, default=32, help='Number of embeddings in VQVAE codebook')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use for training')
    

    return parser.parse_args()

def label_smoothed_cross_entropy(pred, target, smoothing=0.1):
    """
    pred: [B * H * W, C] - logits
    target: [B * H * W] - integer class indices
    """
    C = pred.size(1)  # num classes
    with torch.no_grad():
        # One-hot + smoothing
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing / (C - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    
    log_probs = F.log_softmax(pred, dim=1)
    return F.kl_div(log_probs, true_dist, reduction='batchmean')

def train_model(args):
    # Create unique run name with timestamp
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize wandb
    wandb.init(
        project="VQVAE_mlp_module",
        name=run_name,
        group=f"experiment_{args.epochs}_lr_{args.lr}",
        config=vars(args)
    )
    
    # Define metrics to track
    wandb.define_metric("train/epoch_loss", summary="mean", step_metric="epoch")
    wandb.define_metric("val/epoch_loss", summary="mean", step_metric="epoch")
    wandb.define_metric("train/batch_loss", summary="mean", step_metric="train/batch")
    wandb.define_metric("train/learning_rate", summary="mean", step_metric="epoch")
    wandb.define_metric("train/nll_loss", summary="mean", step_metric="epoch")
    wandb.define_metric("val/nll_loss", summary="mean", step_metric="epoch")
    
    # Early stopping parameters
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    
    # Create save directory with timestamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.save_dir, current_time)
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dataloaders using VITMLP specific loader
    train_loader, val_loader = get_vitmlp_dataloaders(args)
    
    # Initialize model
    args.vit_checkpoint = args.vit_checkpoint
    args.vqvae_checkpoint = args.vqvae_checkpoint
    args.out_size = 32
    args.small_size = 16
    args.n_embeddings = args.n_embeddings
    args.device = args.device
    model = VITMLP(args)
    
    # Log model architecture
    wandb.watch(model.mlp_prior, log="all")
    
    # Only MLP prior needs training
    optimizer = optim.AdamW(model.mlp_prior.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    criterion = label_smoothed_cross_entropy
    nll_criterion = nn.NLLLoss()
    
    for epoch in range(args.epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
            
        model.train()
        train_loss = 0
        train_losses = []
        train_nll_losses = []
        
        # Training loop
        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            images = images.to(args.device)
            
            # Get VQVAE encoding targets
            with torch.no_grad():
                targets = model.get_vqvae_latents(images)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(images)
            
            logits = logits.permute(0, 2, 3, 1).reshape(-1, args.n_embeddings)
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            nll_loss = nll_criterion(logits.log_softmax(dim=-1), targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.mlp_prior.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_losses.append(loss.item())
            train_nll_losses.append(nll_loss.item())
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_nll_loss": nll_loss.item(),
                    "train/batch": epoch * len(train_loader) + batch_idx,
                    "epoch": epoch + 1,
                    "train/learning_rate": scheduler.get_last_lr()[0]
                })
        
        # Validation
        model.eval()
        val_loss = 0
        val_losses = []
        val_nll_losses = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(args.device)
                targets = model.get_vqvae_latents(images)
                logits = model(images)
                logits = logits.view(-1, args.n_embeddings)
                targets = targets.view(-1)
                batch_val_loss = criterion(logits, targets).item()
                batch_val_nll_loss = nll_criterion(logits.log_softmax(dim=-1), targets).item()
                val_loss += batch_val_loss
                val_losses.append(batch_val_loss)
                val_nll_losses.append(batch_val_nll_loss)
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_nll_loss = sum(train_nll_losses) / len(train_nll_losses)
        val_nll_loss = sum(val_nll_losses) / len(val_nll_losses)
        current_lr = scheduler.get_last_lr()[0]
        
        # Print stats
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train NLL: {train_nll_loss:.4f}, Val NLL: {val_nll_loss:.4f}, LR: {current_lr:.2e}")
        
        # Log epoch metrics
        wandb.log({
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "train/nll_loss": train_nll_loss,
            "val/nll_loss": val_nll_loss,
            "train/learning_rate": current_lr,
            "epoch": epoch + 1,
            "train/loss_std": np.std(train_losses),
            "val/loss_std": np.std(val_losses),
            "train/loss_min": np.min(train_losses),
            "val/loss_min": np.min(val_losses),
            "train/loss_max": np.max(train_losses),
            "val/loss_max": np.max(val_losses)
        })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }, checkpoint_path)
            # Log best model
            wandb.save(checkpoint_path)
            # Log best metrics
            wandb.log({
                "best/val_loss": val_loss,
                "best/train_loss": train_loss,
                "best/epoch": epoch + 1
            })
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
        
if __name__ == '__main__':
    args = parse_args()
    train_model(args)
