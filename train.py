
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os

from model.recurrent_denoiser import RecurrentDenoiser

class SpectrogramDataset(Dataset):
    """Custom Dataset for loading spectrogram pairs."""
    def __init__(self, metadata_path):
        self.metadata = pd.read_csv(metadata_path)
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        noisy_path = self.metadata.iloc[idx]['noisy_spectrogram_path']
        clean_path = self.metadata.iloc[idx]['clean_spectrogram_path']
        
        noisy_spec = np.load(noisy_path).T # Transpose to (seq_len, features)
        clean_spec = np.load(clean_path).T
        
        return torch.FloatTensor(noisy_spec), torch.FloatTensor(clean_spec)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataset = SpectrogramDataset(os.path.join(args.data_dir, 'metadata.csv'))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model
    # Input size is half of N_FFT + 1
    input_size = (512 // 2) + 1
    model = RecurrentDenoiser(
        input_size=input_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print("--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for noisy, clean in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(noisy)
            
            loss = criterion(outputs, clean)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'denoiser.pth'))
    print("Training complete. Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the audio denoising model.")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
    args = parser.parse_args()
    main(args)
