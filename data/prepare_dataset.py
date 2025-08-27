
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import soundfile as sf

# Assuming utils.audio_processing is in the parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.audio_processing import audio_to_spectrogram

def find_audio_files(directory, extensions=['.wav', '.flac']):
    """Recursively find all audio files in a directory."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def mix_audio(clean_file, noise_file, snr_db):
    """Mixes a clean audio file with a noise file at a specific SNR."""
    clean_audio, sr = librosa.load(clean_file, sr=None)
    noise_audio, _ = librosa.load(noise_file, sr=sr)

    # Ensure noise is at least as long as clean audio
    if len(noise_audio) < len(clean_audio):
        repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)
    noise_audio = noise_audio[:len(clean_audio)]

    # Calculate powers and scale noise
    clean_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noise_audio ** 2)
    snr = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (snr * noise_power))
    
    noisy_audio = clean_audio + noise_audio * scale
    return clean_audio, noisy_audio, sr

def main(args):
    """Main function to create and save the dataset."""
    clean_files = find_audio_files(args.clean_dir)
    noise_files = find_audio_files(args.noise_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    
    metadata = []
    
    print(f"Creating dataset with {args.num_samples} samples...")
    for i in tqdm(range(args.num_samples)):
        clean_file = np.random.choice(clean_files)
        noise_file = np.random.choice(noise_files)
        snr_db = np.random.uniform(args.min_snr, args.max_snr)
        
        clean_audio, noisy_audio, sr = mix_audio(clean_file, noise_file, snr_db)
        
        # Convert to spectrograms
        clean_mag, _ = audio_to_spectrogram(clean_audio)
        noisy_mag, _ = audio_to_spectrogram(noisy_audio)
        
        # Save spectrograms as numpy files
        clean_spec_path = os.path.join(args.output_dir, f"clean_{i}.npy")
        noisy_spec_path = os.path.join(args.output_dir, f"noisy_{i}.npy")
        np.save(clean_spec_path, clean_mag)
        np.save(noisy_spec_path, noisy_mag)
        
        metadata.append([clean_spec_path, noisy_spec_path, snr_db])

    # Save metadata
    df = pd.DataFrame(metadata, columns=['clean_spectrogram_path', 'noisy_spectrogram_path', 'snr_db'])
    df.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)
    
    print(f"Dataset created successfully in '{args.output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare audio dataset for denoising.")
    parser.add_argument('--clean_dir', type=str, required=True, help='Directory with clean audio files.')
    parser.add_argument('--noise_dir', type=str, required=True, help='Directory with noise audio files.')
    parser.add_argument('--output_dir', type=str, default='processed_data', help='Directory to save processed data.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of noisy samples to generate.')
    parser.add_argument('--min_snr', type=int, default=-5, help='Minimum SNR in dB.')
    parser.add_argument('--max_snr', type=int, default=15, help='Maximum SNR in dB.')
    
    args = parser.parse_args()
    main(args)
