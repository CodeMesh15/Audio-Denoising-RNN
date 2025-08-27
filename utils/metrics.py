
import numpy as np

def calculate_snr(clean_signal, noisy_signal):
    """Calculates the SNR between a clean and a noisy signal."""
    noise = noisy_signal - clean_signal
    
    power_clean = np.sum(clean_signal ** 2)
    power_noise = np.sum(noise ** 2)
    
    if power_noise == 0:
        return float('inf')
        
    snr = 10 * np.log10(power_clean / power_noise)
    return snr
