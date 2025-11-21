import os
import torch
import pandas as pd
from vit_mlp import VITMLP
from torchvision.io import read_image
from torchvision.transforms import Resize
import argparse

def load_model(vit_checkpoint, vqvae_checkpoint, device='cuda'):
    """Load trained VAExViT model"""
    model = VITMLP(
        vit_checkpoint=vit_checkpoint,
        vqvae_checkpoint=vqvae_checkpoint,
        device=device
    )
    model.eval()
    return model

def evaluate_test_set(model, test_dir, threshold=0.9987):
    """Evaluate model on test set and calculate accuracy metrics"""
    results = []
    resize = Resize((256, 256))
    
    # Process all images in test directory
    for img_file in os.listdir(test_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Determine label from filename prefix
        true_label = 0 if img_file.lower().startswith('normal') else 1
        img_path = os.path.join(test_dir, img_file)
        try:
            # Load and convert grayscale to RGB by repeating channels
            img = read_image(img_path).float() / 255.0
            if img.shape[0] == 1:  # If grayscale
                img = img.repeat(3, 1, 1)  # Convert to RGB
            print(f"Original image shape: {img.shape}")  # Debug
            
            img = resize(img).to(model.device)
            print(f"Resized image shape: {img.shape}")  # Debug
            
            img = img.unsqueeze(0)  # Add batch dimension
            print(f"Batch image shape: {img.shape}")  # Debug
            
            # Convert to ViT expected format [batch, num_patches, patch_dim]
            b, c, h, w = img.shape
            print(f"Input shape - batch:{b} channels:{c} height:{h} width:{w}")  # Debug
            
            patch_size = 32
            n_patches = (h // patch_size) * (w // patch_size)
            patch_dim = c * patch_size * patch_size
            print(f"Calculated - n_patches:{n_patches} patch_dim:{patch_dim}")  # Debug
            
            # Extract patches and flatten
            patches = img.unfold(2, patch_size, patch_size)  # [b,c,h/patch_size,w,patch_size]
            patches = patches.unfold(3, patch_size, patch_size)  # [b,c,h/patch_size,w/patch_size,patch_size,patch_size]
            patches = patches.contiguous().view(b, n_patches, -1)  # [b,n_patches,patch_dim]
            print(f"Final patches shape: {patches.shape}")  # Debug
            
            # Get prediction and debug info
            score, pred_label = model.predict_anomaly(img, threshold)
            print(f"Prediction - score: {score.item():.4f}, label: {pred_label.item()}, true: {true_label}")  # Debug
            
            results.append({
                'image': img_file,
                'true_label': true_label,
                'pred_label': pred_label.item(),
                'score': score.item()
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Calculate metrics if we have results
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("\nNo valid test images found!")
        return df
        
    try:
        accuracy = (df['true_label'] == df['pred_label']).mean()
        precision = df[df['pred_label'] == 1]['true_label'].mean()
        recall = df[df['true_label'] == 1]['pred_label'].mean()
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nTest Results (threshold={threshold:.4f}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}") 
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    except KeyError as e:
        print(f"\nError calculating metrics: {str(e)}")
        print("Please check your test directory structure - it should contain 'normal' and 'anomalous' subfolders")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_checkpoint', type=str, 
                       default='VisionTransformer/checkpoints/best_model.pth')
    parser.add_argument('--vqvae_checkpoint', type=str,
                       default='VQVAE/saved_models/2025-05-10_02-03-50/best-epoch=27-val_total_loss=0.0495.ckpt')
    parser.add_argument('--test_dir', type=str, 
                       default='VQVAE/test_dataset',
                       help='Directory with normal/anomalous subfolders')
    parser.add_argument('--threshold', type=float,
                       default=0.9987,
                       help='Anomaly detection threshold')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.vit_checkpoint, args.vqvae_checkpoint, device)
    
    print(f"Evaluating on test set: {args.test_dir}")
    results_df = evaluate_test_set(model, args.test_dir, args.threshold)
    
    # Save results
    output_file = os.path.join(args.test_dir, 'test_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
