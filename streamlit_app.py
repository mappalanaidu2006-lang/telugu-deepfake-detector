import os
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
from spafe.features.lfcc import lfcc

# Set page config
st.set_page_config(
    page_title="Telugu Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered"
)

# App Header
st.title("🎙️ Telugu Deepfake Audio Detector")
st.markdown("""
Detect whether a Telugu audio recording is **Real Human Speech** or **AI-Generated (Fake)**.
* **Model:** CNN + Bi-LSTM (Deep Learning)
* **Features:** Fused MFCC (39) + LFCC (39) with CMVN Normalization
* **Base Paper:** Owais et al., IEEE Access 2026
---
""")

MODEL_PATH = "deepfake_audio_cnn_bilstm.keras"
SR = 16000
DURATION = 4
TARGET_LEN = SR * DURATION
N_CEPS = 13
TARGET_FRAMES = 400

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    elif os.path.exists(r"D:\projects\Dataset\Features\deepfake_audio_cnn_bilstm.keras"):
        return tf.keras.models.load_model(r"D:\projects\Dataset\Features\deepfake_audio_cnn_bilstm.keras")
    else:
        st.error("Model file not found!")
        return None

model = load_model()

def extract_features(audio):
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed) > 0:
        audio = trimmed

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    if len(audio) > TARGET_LEN:
        audio = audio[:TARGET_LEN]
    elif len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)), mode="constant")

    m_stat = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_CEPS, n_fft=512, hop_length=160)
    m_d1   = librosa.feature.delta(m_stat)
    m_d2   = librosa.feature.delta(m_stat, order=2)
    mfcc   = np.vstack([m_stat, m_d1, m_d2])

    try:
        l_raw = lfcc(sig=audio, fs=SR, num_ceps=N_CEPS, nfilts=26)
    except TypeError:
        l_raw = lfcc(sig=audio, fs=SR, num_ceps=N_CEPS)

    l_stat = l_raw.T
    l_d1   = librosa.feature.delta(l_stat)
    l_d2   = librosa.feature.delta(l_stat, order=2)
    lfcc_f = np.vstack([l_stat, l_d1, l_d2])

    def fix(mat):
        if mat.shape[1] > TARGET_FRAMES:
            return mat[:, :TARGET_FRAMES]
        elif mat.shape[1] < TARGET_FRAMES:
            return np.pad(mat, ((0, 0), (0, TARGET_FRAMES - mat.shape[1])), mode="edge")
        return mat

    mfcc   = fix(mfcc)
    lfcc_f = fix(lfcc_f)

    combined = np.vstack([mfcc, lfcc_f]).T
    mean     = np.mean(combined, axis=0, keepdims=True)
    std      = np.std(combined,  axis=0, keepdims=True) + 1e-8
    norm     = (combined - mean) / std
    return np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=-1.0)

# File uploader
uploaded_file = st.file_uploader("Upload a Telugu Audio File (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("🔍 Analyze Audio", type="primary"):
        with st.spinner("Analyzing audio features..."):
            audio, sr = librosa.load(uploaded_file, sr=SR, mono=True)
            features = extract_features(audio)
            input_tensor = np.expand_dims(features, axis=0)

            fake_prob = float(model.predict(input_tensor, verbose=0)[0][0])
            real_prob = 1.0 - fake_prob

            fake_pct = fake_prob * 100.0
            real_pct = real_prob * 100.0

            st.subheader("📊 Detection Results")
            
            if fake_prob >= 0.50:
                st.error(f"🔴 **FAKE AUDIO DETECTED** (Confidence: {fake_pct:.2f}%)")
            else:
                st.success(f"🟢 **REAL AUDIO DETECTED** (Confidence: {real_pct:.2f}%)")

            col1, col2 = st.columns(2)
            col1.metric("Real Probability", f"{real_pct:.2f}%")
            col2.metric("Fake Probability", f"{fake_pct:.2f}%")