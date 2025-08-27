import streamlit as st
import torch
import librosa
import soundfile as sf
import numpy as np
import os
import tempfile
from model.recurrent_denoiser import RecurrentDenoiser
from utils.audio_processing import audio_to_spectrogram, spectrogram_to_audio

# Set page configuration
st.set_page_config(
    page_title="Audio Denoising RNN",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #4b6cb7, #182848);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .section-header {
        color: #4b6cb7;
        border-bottom: 2px solid #4b6cb7;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    .stButton>button {
        background-color: #4b6cb7;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #182848;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .audio-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #4b6cb7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main header with gradient background
st.markdown("""
<div class="main-header">
    <h1>🎵 Audio Denoising with RNN</h1>
    <p style="font-size: 1.2rem;">Remove noise from your audio files using Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h2 class='section-header'>Upload & Process</h2>", unsafe_allow_html=True)
    
    # File uploader with better styling
    uploaded_file = st.file_uploader("Choose a noisy audio file", type=["wav", "mp3", "flac", "m4a"])
    
    if uploaded_file is not None:
        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
            tmp_input.write(uploaded_file.getvalue())
            input_path = tmp_input.name

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        # Load and display original audio
        st.markdown("<h3>Original (Noisy) Audio</h3>", unsafe_allow_html=True)
        st.markdown("<div class='audio-container'>", unsafe_allow_html=True)
        st.audio(input_path)
        st.markdown("</div>", unsafe_allow_html=True)

        # Process button
        if st.button("✨ Denoise Audio"):
            with st.spinner("Processing audio... This may take a few seconds"):
                try:
                    # Check if model exists
                    model_path = "denoiser.pth"
                    if not os.path.exists(model_path):
                        st.error(f"Model file not found at {model_path}. Please check the path or train a model first.")
                        st.stop()

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    # Load model with fixed parameters that match the trained model
                    input_size = (512 // 2) + 1  # 257 (fixed)
                    hidden_size = 256  # Must match the trained model
                    num_layers = 2     # Must match the trained model
                    
                    model = RecurrentDenoiser(
                        input_size=input_size, 
                        hidden_size=hidden_size,
                        num_layers=num_layers
                    ).to(device)
                    
                    # Load model weights
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()

                    # Load audio
                    noisy_audio, sr = librosa.load(input_path, sr=None)
                    
                    # Process
                    noisy_mag, noisy_phase = audio_to_spectrogram(noisy_audio)
                    noisy_mag_t = torch.FloatTensor(noisy_mag.T).unsqueeze(0).to(device)

                    with torch.no_grad():
                        denoised_mag_t = model(noisy_mag_t)
                        
                    denoised_mag = denoised_mag_t.squeeze(0).cpu().numpy().T
                    
                    # Reconstruct audio
                    denoised_audio = spectrogram_to_audio(denoised_mag, noisy_phase)
                    
                    # Save denoised audio
                    sf.write(output_path, denoised_audio, sr)
                    
                    # Display results
                    st.markdown("<h3>✅ Denoised Audio</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='audio-container'>", unsafe_allow_html=True)
                    st.audio(output_path)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Provide download button
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Denoised Audio",
                            data=file,
                            file_name="denoised_audio.wav",
                            mime="audio/wav",
                            key="download"
                        )
                    
                    st.markdown("<div class='success-box'><strong>🎉 Success!</strong> Audio denoising completed successfully!</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"<div class='error-box'><strong>❌ Error:</strong> {str(e)}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='warning-box'><strong>💡 Solution:</strong> The model parameters must match the trained model. Using default parameters (Hidden Size: 256, Layers: 2).</div>", unsafe_allow_html=True)
            
            # Clean up temporary files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass
    else:
        st.info("👆 Please upload an audio file to begin denoising")

with col2:
    st.markdown("<h2 class='section-header'>Model Information</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Default Model Architecture:</strong><br>
        • Input Size: 257 (STFT bins)<br>
        • Hidden Size: 256 (GRU units)<br>
        • Layers: 2 (Stacked GRU)<br>
        • Output: Linear Layer<br><br>
        <em>Note: To use different parameters, you need to train a new model with those specifications.</em>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-header'>Sample Audio</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Try the app with this sample noisy audio file:</div>", unsafe_allow_html=True)
    
    if os.path.exists("sample_noisy_audio.wav"):
        st.audio("sample_noisy_audio.wav")
        with open("sample_noisy_audio.wav", "rb") as file:
            st.download_button(
                label="📥 Download Sample",
                data=file,
                file_name="sample_noisy_audio.wav",
                mime="audio/wav",
                key="sample_download"
            )
    else:
        st.info("Sample audio file not found. Upload your own file to test.")
    
    st.markdown("<h2 class='section-header'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Process:</strong><br>
        1. Audio → Spectrogram (STFT)<br>
        2. Noisy spectrogram → RNN<br>
        3. RNN → Clean spectrogram<br>
        4. Clean spectrogram → Audio (ISTFT)
    </div>
    """, unsafe_allow_html=True)

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Audio Denoising RNN</strong> - Powered by Deep Learning</p>
    <p>Upload noisy audio files and get cleaner versions using Recurrent Neural Networks</p>
</div>
""", unsafe_allow_html=True)