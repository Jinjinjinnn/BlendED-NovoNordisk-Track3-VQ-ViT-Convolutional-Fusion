import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from vit_mlp import VITMLP
import argparse
import glob
from sklearn.metrics import roc_curve, auc
import pandas as pd
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_checkpoint', type=str, required=True,
                       help='Path to ViT checkpoint')
    parser.add_argument('--vqvae_checkpoint', type=str, required=True,
                       help='Path to VQVAE checkpoint')
    parser.add_argument('--mlp_checkpoint', type=str, default=None,
                       help='Path to MLP checkpoint (optional)')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--threshold', type=float, default=0.9987,
                       help='Anomaly detection threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on')
    parser.add_argument('--save_dir', type=str, default='inference_results',
                       help='Directory to save result images')
    return parser.parse_args()

# Configuration
IMAGE_SIZE = 512

def remove_margin(pil_img, threshold=240):
    img_np = np.array(pil_img)
    mask = img_np < threshold
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return pil_img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img_np[y0:y1, x0:x1]
    return Image.fromarray(cropped)

# 이미지 전처리 파이프라인 (512x512, grayscale, margin 제거, ToTensor, Normalize)
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # 1채널 그레이스케일
    img = remove_margin(img, threshold=240)
    img = img.resize((512, 512))
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    img = torch.from_numpy(img).unsqueeze(0)  # (1, 512, 512)
    return img

def load_separate_models(vit_checkpoint, vqvae_checkpoint, mlp_checkpoint=None, device='cuda'):
    """Load VAExViT model with separate checkpoints for each component"""
    # Create args object
    class Args:
        def __init__(self):
            self.vit_checkpoint = vit_checkpoint
            self.vqvae_checkpoint = vqvae_checkpoint
            self.device = device
            self.small_size = 16
            self.out_size = 32
            self.n_embeddings = 32
    
    args = Args()
    
    # Initialize model with args
    model = VITMLP(args)
    
    # Load MLP weights if provided
    if mlp_checkpoint:
        mlp_state = torch.load(mlp_checkpoint, map_location=device)
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
            model.mlp_prior.load_state_dict(mlp_state_dict)
    
    model.eval()
    return model

def load_combined_model(checkpoint_path, device='cuda'):
    """Load VAExViT model from combined checkpoint (legacy support)"""
    # Create args object
    class Args:
        def __init__(self):
            self.vit_checkpoint = checkpoint_path
            self.vqvae_checkpoint = checkpoint_path
            self.device = device
            self.small_size = 16
            self.out_size = 32
            self.n_embeddings = 32
    
    args = Args()
    
    # Initialize model with args
    model = VITMLP(args)
    model.eval()
    return model

def predict_image(model, image_path, threshold):
    img_tensor = preprocess_image(image_path).unsqueeze(0).to(model.device)  # [B, 1, 512, 512]

    with torch.no_grad():
        # Prepare ViT input
        vit_input = img_tensor.repeat(1, 3, 1, 1)  # [B, 3, 512, 512]
        _, probs = model(vit_input, return_probs=True)  # [B, n_embed, H, W]

        # Predict latent indices from ViT
        encoding_indices = torch.argmax(probs, dim=1)  # [B, H, W]
        flat_indices = encoding_indices.view(-1)

        # Quantize from codebook
        codebook = model.vqvae.vector_quantization.embedding
        quantized = codebook(flat_indices)  # [B*H*W, C]
        embedding_dim = quantized.shape[1]
        B, H, W = encoding_indices.shape
        quantized = quantized.view(B, H, W, embedding_dim).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Decode to reconstruction
        recon = model.vqvae.decoder(quantized)  # [B, 1, 512, 512]

        # Get ALM (Alignment Loss Map) from VQVAE
        _, _, _, alm_map, _, _, _ = model.vqvae(img_tensor, return_extra=True)  # [B, 1, H, W]
        alm_up = F.interpolate(alm_map, size=(512, 512), mode='bilinear', align_corners=False)

        # Get NLL
        gt_indices = model.get_vqvae_latents(img_tensor)  # [B, H, W]
        selected_probs = torch.gather(probs, 1, gt_indices.unsqueeze(1)).squeeze(1)
        nll = -torch.log(selected_probs + 1e-6)
        nll_up = F.interpolate(nll.unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)

        # Final anomaly map
        anomaly_map = (alm_up + nll_up) / 2
        score = anomaly_map.mean().item()
        label = 'Anomaly' if score > threshold else 'Normal'

    # Visualization prep
    orig_img = Image.open(image_path).convert('L').resize((512, 512))
    orig_img = np.array(orig_img)

    def to_np(t):
        t = t.squeeze().cpu().numpy()
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    return {
        'image': orig_img,
        'reconstruction': to_np(recon),
        'alm': to_np(alm_up),
        'nll': to_np(nll_up),
        'anomaly_map': to_np(anomaly_map),
        'score': score,
        'label': label
    }


def visualize_results(results, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)

    # 1. Original Image
    axes[0].imshow(results['image'], cmap='gray')
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # 2. Reconstruction
    axes[1].imshow(results['reconstruction'], cmap='gray')
    axes[1].set_title('Reconstruction', fontsize=14)
    axes[1].axis('off')

    # 3. ALM Map
    axes[2].imshow(results['alm'], cmap='jet', aspect='auto')
    axes[2].set_title('ALM Map', fontsize=14)
    axes[2].axis('off')

    # 4. NLL Map
    # ✅ Normalize and clip for better visual range
    nll = results['nll']
    nll_clipped = np.clip(nll, 0, np.percentile(nll, 98))  # suppress outliers
    nll_norm = (nll_clipped - nll_clipped.min()) / (nll_clipped.max() - nll_clipped.min() + 1e-8)

    heat = axes[3].imshow(nll_norm, cmap='jet', aspect='auto')
    axes[3].set_title(f'NLL Map (Score: {results["score"]:.4f})', fontsize=14)
    axes[3].axis('off')
    cbar = plt.colorbar(heat, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_test_set(model, test_dir, threshold, save_dir=None):
    """Evaluate all images in test directory and analyze scores, save result images if save_dir 지정"""
    # Get all test images
    normal_paths = glob.glob(f"{test_dir}/NORMAL-*.jpeg")
    anomaly_paths = glob.glob(f"{test_dir}/*-*.jpeg")  # Gets all other classes
    
    # 디렉토리 생성
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Collect scores and labels
    scores = []
    labels = []
    results = []
    
    # Process normal images
    for path in normal_paths:
        res = predict_image(model, path, threshold)
        scores.append(res['score'])
        labels.append(0)  # 0 = normal
        results.append((path, res['score'], 'Normal'))
        # 결과 이미지 저장
        if save_dir is not None:
            fname = os.path.splitext(os.path.basename(path))[0] + '_result.png'
            save_path = os.path.join(save_dir, fname)
            visualize_results(res, save_path=save_path)
    
    # Process anomaly images
    for path in anomaly_paths:
        if 'NORMAL' not in path:  # Skip normals again
            res = predict_image(model, path, threshold)
            scores.append(res['score'])
            labels.append(1)  # 1 = anomaly
            results.append((path, res['score'], 'Anomaly'))
            # 결과 이미지 저장
            if save_dir is not None:
                fname = os.path.splitext(os.path.basename(path))[0] + '_result.png'
                save_path = os.path.join(save_dir, fname)
                visualize_results(res, save_path=save_path)
    
    # Create DataFrame of results
    df = pd.DataFrame(results, columns=['path', 'score', 'label'])
    
    # Plot score distributions & ROC curve만 남김
    plt.figure(figsize=(6, 5))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()
    else:
        plt.show()
    # Score만 출력
    print(f"Scores: {scores}")
    # Print optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    return df

if __name__ == '__main__':
    args = parse_args()
    
    # Initialize model with separate weights
    model = load_separate_models(
        vit_checkpoint=args.vit_checkpoint,
        vqvae_checkpoint=args.vqvae_checkpoint,
        mlp_checkpoint=args.mlp_checkpoint,
        device=args.device
    )
    
    # Evaluate on test set
    results_df = evaluate_test_set(model, args.test_dir, args.threshold, save_dir=args.save_dir) 