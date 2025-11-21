import torch
import sys
import os

def check_vit_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"- {key}")
    
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print("\nState dict keys:")
    for key in state_dict.keys():
        print(f"- {key}")
    
    # Check if keys start with 'vit.'
    vit_keys = [k for k in state_dict.keys() if k.startswith('vit.')]
    if vit_keys:
        print("\nFound keys with 'vit.' prefix:")
        for key in vit_keys:
            print(f"- {key}")
        
        # Create new state dict without 'vit.' prefix
        new_state_dict = {k.replace('vit.', ''): v for k, v in state_dict.items() if k.startswith('vit.')}
        print("\nNew state dict keys (without 'vit.' prefix):")
        for key in new_state_dict.keys():
            print(f"- {key}")
        
        # Save modified checkpoint
        modified_checkpoint = {'state_dict': new_state_dict}
        save_path = checkpoint_path.replace('.pth', '_modified.pth')
        torch.save(modified_checkpoint, save_path)
        print(f"\nSaved modified checkpoint to: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_vit_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    check_vit_checkpoint(checkpoint_path) 