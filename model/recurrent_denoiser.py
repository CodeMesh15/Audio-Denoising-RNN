
import torch
import torch.nn as nn

class RecurrentDenoiser(nn.Module):
    """
    A deep GRU-based model for audio denoising.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(RecurrentDenoiser, self).__init__()
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input noisy spectrogram of shape (batch, seq_len, features).
        """
        # Pass through GRU
        gru_out, _ = self.gru(x)
        
        # Pass through the linear layer
        output = self.fc(gru_out)
        
        # We can use a sigmoid to ensure the output mask is between 0 and 1
        # or a ReLU if predicting the spectrogram directly to ensure non-negativity.
        output = torch.relu(output)
        
        return output
