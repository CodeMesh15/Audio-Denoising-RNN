import numpy as np
import os
import torch
import torch.nn as nn
from model.recurrent_denoiser import RecurrentDenoiser

def create_dummy_dataset():
    \"\"\"Create a dummy dataset for testing purposes.\"\"\"
    os.makedirs('dummy_data', exist_ok=True)
    
    # Create dummy spectrograms
    for i in range(10):
        # Create random noisy and clean spectrograms
        seq_len = 100
        features = 257  # (512//2) + 1
        
        noisy_spec = np.random.rand(seq_len, features).astype(np.float32)
        clean_spec = np.random.rand(seq_len, features).astype(np.float32)
        
        # Save as .npy files
        np.save(f'dummy_data/noisy_{i}.npy', noisy_spec.T)  # Transpose for saving
        np.save(f'dummy_data/clean_{i}.npy', clean_spec.T)
    
    # Create a simple metadata file
    with open('dummy_data/metadata.csv', 'w') as f:
        f.write('noisy_spectrogram_path,clean_spectrogram_path\\n')
        for i in range(10):
            f.write(f'dummy_data/noisy_{i}.npy,dummy_data/clean_{i}.npy\\n')
    
    print("Dummy dataset created with 10 samples")

def train_dummy_model():
    \"\"\"Train a model with the dummy dataset for testing.\"\"\"
    os.makedirs('models', exist_ok=True)
    
    # Create dummy dataset
    create_dummy_dataset()
    
    # Model parameters
    input_size = (512 // 2) + 1  # 257
    hidden_size = 128
    num_layers = 2
    batch_size = 2
    epochs = 3
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentDenoiser(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training dummy model for testing...")
    
    # Training loop with dummy data
    for epoch in range(epochs):
        total_loss = 0
        for i in range(5):  # 5 batches
            # Create dummy batch data
            noisy_batch = torch.rand(batch_size, 100, input_size).to(device)
            clean_batch = torch.rand(batch_size, 100, input_size).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(noisy_batch)
            loss = criterion(outputs, clean_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / 5
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/denoiser.pth')
    print("Dummy model saved to models/denoiser.pth")

if __name__ == "__main__":
    train_dummy_model()