import torch
import librosa
import soundfile as sf
import argparse
import numpy as np
import os

from model.recurrent_denoiser import RecurrentDenoiser
from utils.audio_processing import audio_to_spectrogram, spectrogram_to_audio

def denoise_audio(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    input_size = (512 // 2) + 1
    model = RecurrentDenoiser(input_size=input_size, hidden_size=256, num_layers=2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Load audio
    noisy_audio, sr = librosa.load(args.input_file, sr=None)
    
    # Process
    noisy_mag, noisy_phase = audio_to_spectrogram(noisy_audio)
    noisy_mag_t = torch.FloatTensor(noisy_mag.T).unsqueeze(0).to(device) # (1, seq, feat)

    with torch.no_grad():
        denoised_mag_t = model(noisy_mag_t)
        
    denoised_mag = denoised_mag_t.squeeze(0).cpu().numpy().T
    
    # Reconstruct audio
    denoised_audio = spectrogram_to_audio(denoised_mag, noisy_phase)
    
    # Save
    sf.write(args.output_file, denoised_audio, sr)
    print(f"Denoised audio saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Denoise an audio file.")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    
    args = parser.parse_args()
    denoise_audio(args)
