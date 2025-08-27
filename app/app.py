import streamlit as st
import torch
import librosa
import numpy as np
import soundfile as sf
import io

from model.recurrent_denoiser import RecurrentDenoiser
from utils.audio_processing import audio_to_spectrogram, spectrogram_to_audio

# --- Model Loading ---
@st.cache_resource
def load_model(model_path='models/denoiser.pth'):
    """Load the trained PyTorch model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (512 // 2) + 1
    model = RecurrentDenoiser(input_size=input_size, hidden_size=256, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- UI ---
st.set_page_config(layout="wide")
st.title("🎙️ Audio Denoising with a Recurrent Neural Network")

st.write("""
Upload a noisy audio file (.wav, .mp3, .flac) and the model will attempt to remove the background noise.
This app is an implementation of the project described in the resume.
""")

# Load the model
try:
    model, device = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a noisy audio file", type=['wav', 'mp3', 'flac'])

if uploaded_file is not None:
    st.header("Processing Audio")
    
    # Load audio
    try:
        noisy_audio, sr = librosa.load(uploaded_file, sr=None)
        st.write("Original Noisy Audio:")
        st.audio(uploaded_file)
    except Exception as e:
        st.error(f"Could not load audio file: {e}")
        st.stop()

    # Denoise
    with st.spinner("Denoising in progress..."):
        try:
            # Process
            noisy_mag, noisy_phase = audio_to_spectrogram(noisy_audio)
            noisy_mag_t = torch.FloatTensor(noisy_mag.T).unsqueeze(0).to(device)

            with torch.no_grad():
                denoised_mag_t = model(noisy_mag_t)
            
            denoised_mag = denoised_mag_t.squeeze(0).cpu().numpy().T
            
            # Reconstruct audio
            denoised_audio = spectrogram_to_audio(denoised_mag, noisy_phase)

            # Save to an in-memory buffer
            buffer = io.BytesIO()
            sf.write(buffer, denoised_audio, sr, format='WAV')
            buffer.seek(0)
            
            st.write("Denoised Audio:")
            st.audio(buffer, format='audio/wav')
            
            st.success("Denoising complete!")

        except Exception as e:
            st.error(f"An error occurred during denoising: {e}")
