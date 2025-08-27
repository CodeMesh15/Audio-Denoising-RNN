# Audio-Denoising-RNN

An implementation of a Deep Recurrent Gated Neural Network for dynamic audio denoising. The model learns to reconstruct a clean audio signal from a noise-corrupted input by estimating and subtracting the noise in the spectral domain, inspired by research at Carnegie Mellon University.

---

## What This Repository Does For You

This Audio Denoising RNN repository helps you remove unwanted noise from audio files using artificial intelligence. Here's what it specifically does for you:

### Core Functionality
- **Noise Removal**: Takes noisy audio files (like recordings with background hum, traffic, or static) and produces cleaner versions
- **AI-Powered**: Uses a Recurrent Neural Network (RNN) with Gated Recurrent Units (GRUs) to intelligently distinguish between speech/music and noise

### Practical Applications
1. **Voice Recording Enhancement**: Clean up podcast recordings, voice memos, or interview audio
2. **Historical Audio Restoration**: Improve old recordings or tapes with background noise
3. **Video Conferencing**: Reduce fan noise, keyboard typing, or street sounds from remote meeting recordings
4. **Music Production**: Remove unwanted artifacts from live recordings
5. **Research/Data Preparation**: Clean audio data for speech recognition or analysis projects

### Technical Benefits
- **Spectral Processing**: Works in the frequency domain for more precise noise identification
- **Sequence Learning**: Uses RNNs to understand temporal patterns in audio (knows what comes next)
- **Customizable**: You can train it on specific types of noise you commonly encounter

### What You Get
- A ready-to-use denoising tool that works immediately with the provided test setup
- Complete training pipeline if you want to improve results for your specific noise types
- Educational example of deep learning applied to audio signal processing

Essentially, it's like having a smart "audio cleaner" that can automatically remove background noise while preserving the important content of your recordings.

---

## 1. Project Overview

This project tackles the classic problem of audio denoising, which is the task of removing unwanted noise from an audio signal. The approach is based on the concept of **dynamic spectral subtraction**, where a neural network learns to estimate a spectral mask or, more directly, the clean audio spectrum itself. This mask, when applied to the noisy signal's spectrum, suppresses the noise components and enhances the clean signal components. We will implement a **Deep Recurrent Gated Neural Network** (like an LSTM or GRU) to perform this signal reconstruction, as mentioned in the project description.

---

## 2. Core Objectives

-   To build a data processing pipeline for creating pairs of noisy and clean audio samples.
-   To implement a Deep Recurrent Gated Neural Network (e.g., GRU or LSTM) for a sequence-to-sequence audio task.
-   To train the model to predict a clean spectrogram from a noisy one.
-   To evaluate the model's performance using audio quality metrics like SNR (Signal-to-Noise Ratio).

---

## 3. Methodology

#### Phase 1: Data Preparation

1.  **Dataset**: We need a dataset of clean speech and a separate dataset of various noise types.
    -   **Clean Speech**: The [LibriSpeech](https://www.openslr.org/12) dataset is a standard choice.
    -   **Noise**: The [DEMAND dataset](https://zenodo.org/record/1227121) or the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset provide a good variety of noise samples.
2.  **Creating Training Samples**:
    -   Load a clean audio file and a random noise file.
    -   Mix them at a random Signal-to-Noise Ratio (SNR) level (e.g., between -5dB and 15dB) to create a noisy audio sample. This pair (`noisy_audio`, `clean_audio`) is our training example.
3.  **Feature Extraction (Spectrograms)**:
    -   Audio is processed in the frequency domain. We will convert all audio signals (both noisy and clean) into their **spectrogram** representations using the **Short-Time Fourier Transform (STFT)**.
    -   The model's input will be the magnitude spectrogram of the noisy audio, and the target will be the magnitude spectrogram of the clean audio.

#### Phase 2: Model Architecture (Deep Recurrent Gated NN)

1.  **Model Choice**: We will implement a model using **Gated Recurrent Units (GRUs)**, which are a type of "gated" RNN mentioned in the resume.
2.  **Architecture**:
    -   The input to the model at each time step is a single frame (a frequency vector) from the noisy spectrogram.
    -   A stack of two or three GRU layers processes the sequence of frames, capturing temporal dependencies in the audio.
    -   A final fully connected (`Linear`) layer maps the output of the last GRU layer to predict the corresponding frame of the clean spectrogram.

#### Phase 3: Training

1.  **Loss Function**: We will use a standard regression loss like **Mean Squared Error (MSE)** between the predicted clean spectrogram and the actual clean spectrogram.
2.  **Training Loop**:
    -   For each batch, the model receives the noisy spectrograms.
    -   It outputs the predicted clean spectrograms.
    -   The MSE loss is calculated and used to update the model's weights.

#### Phase 4: Evaluation and Denoising

1.  **Evaluation Metrics**: We'll use the **Signal-to-Noise Ratio (SNR)** to measure the quality of the denoised audio. A higher SNR means better performance.
2.  **Denoising a new file (`denoise.py`)**:
    -   Load a new noisy audio file.
    -   Convert it to a spectrogram and save its phase information.
    -   Feed the magnitude spectrogram to the trained model to get the predicted clean magnitude spectrogram.
    -   Combine the predicted magnitude with the original phase information.
    -   Use the **Inverse Short-Time Fourier Transform (ISTFT)** to convert the clean spectrogram back into a time-domain audio signal.
    -   Save the resulting denoised audio as a `.wav` file.

---