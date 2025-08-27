import numpy as np
import soundfile as sf
import librosa
import os

def generate_test_audio():
    # Create a directory for test audio
    os.makedirs('test_audio', exist_ok=True)
    
    # Generate a simple test signal (a sine wave)
    duration = 5  # seconds
    sample_rate = 22050  # Hz
    frequency = 440  # Hz (A4 note)
    
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate clean sine wave
    clean_audio = np.sin(2 * np.pi * frequency * t)
    
    # Generate noise
    noise = np.random.normal(0, 0.1, clean_audio.shape)
    
    # Create noisy audio by adding noise to clean audio
    noisy_audio = clean_audio + noise
    
    # Normalize to [-1, 1]
    clean_audio = clean_audio / np.max(np.abs(clean_audio))
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    
    # Save the audio files
    sf.write('test_audio/clean.wav', clean_audio, sample_rate)
    sf.write('test_audio/noisy.wav', noisy_audio, sample_rate)
    
    print("Test audio files generated:")
    print("- test_audio/clean.wav")
    print("- test_audio/noisy.wav")

if __name__ == "__main__":
    generate_test_audio()