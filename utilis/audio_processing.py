
import librosa
import numpy as np

N_FFT = 512
HOP_LENGTH = 256

def audio_to_spectrogram(audio):
    """Converts audio waveform to a magnitude and phase spectrogram."""
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, phase = librosa.magphase(stft)
    return magnitude, phase

def spectrogram_to_audio(magnitude, phase):
    """Converts a magnitude and phase spectrogram back to an audio waveform."""
    stft_matrix = magnitude * phase
    audio = librosa.istft(stft_matrix, hop_length=HOP_LENGTH)
    return audio
