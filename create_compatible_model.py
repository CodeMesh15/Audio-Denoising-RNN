import torch
import os
from model.recurrent_denoiser import RecurrentDenoiser

def create_compatible_model():
    """Create and save a model with the correct parameters."""
    os.makedirs('models', exist_ok=True)
    
    # Model parameters (matching what the app expects)
    input_size = (512 // 2) + 1  # 257
    hidden_size = 256  # This is what the app uses
    num_layers = 2     # This is what the app uses
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentDenoiser(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    
    # Save model
    torch.save(model.state_dict(), 'models/denoiser.pth')
    print("Compatible model saved to models/denoiser.pth")
    
    # Also save to the root directory for easier access
    torch.save(model.state_dict(), 'denoiser.pth')
    print("Compatible model also saved to denoiser.pth")

if __name__ == "__main__":
    create_compatible_model()