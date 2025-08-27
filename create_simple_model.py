import torch
import os
from model.recurrent_denoiser import RecurrentDenoiser

def create_simple_model():
    """Create and save a simple model for testing."""
    os.makedirs('models', exist_ok=True)
    
    # Model parameters (matching what denoise.py expects)
    input_size = (512 // 2) + 1  # 257
    hidden_size = 256  # Changed from 128 to 256 to match denoise.py
    num_layers = 2
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentDenoiser(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    
    # Save model
    torch.save(model.state_dict(), 'models/denoiser.pth')
    print("Simple model saved to models/denoiser.pth")

if __name__ == "__main__":
    create_simple_model()