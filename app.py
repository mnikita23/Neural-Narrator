import os
import numpy as np
import torch
import torchaudio
import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from googletrans import Translator

# Constants
SAMPLE_RATE = 16000
RECORD_DURATION = 5  # seconds

# --- Model Loading ---
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

processor, model = load_model()

# --- Audio Recording ---
def record_audio(duration=RECORD_DURATION, sample_rate=SAMPLE_RATE):
    """Record audio from microphone"""
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()  # Wait until recording is finished
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    sf.write(filename, recording, sample_rate)
    return filename

# --- Audio Processing ---
def preprocess_audio(file_path):
    """Load and prepare audio for model"""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # Ensure waveform is 1D

def predict_transcription(waveform):
    """Convert audio to text"""
    inputs = processor(waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

# --- Translation ---
def translate_text(text, dest_language='hi'):
    """Translate text to the specified language using googletrans"""
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

# --- Streamlit App ---
def main():
    st.title("🎙 Neural Narrator - Live Recording (Phase 2)")
    
    # Initialize session state for recorded file
    if 'recorded_file' not in st.session_state:
        st.session_state.recorded_file = None

    # Tab interface
    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
    
    with tab1:
        # Live Recording Section
        if st.button("Start Recording"):
            st.session_state.recorded_file = record_audio()
            st.audio(st.session_state.recorded_file)
            
        if st.session_state.recorded_file:
            if st.button("Transcribe Recording"):
                waveform = preprocess_audio(st.session_state.recorded_file)
                try:
                    transcription = predict_transcription(waveform)
                    st.subheader("Transcription:")
                    st.success(transcription)

                    # Translate the transcription to Hindi
                    translated_text = translate_text(transcription)
                    st.subheader("Translation to Hindi:")
                    st.success(translated_text)

                except Exception as e:
                    st.error(f"Error during transcription: {e}")
                # Optionally, keep the file for further use
                # os.remove(st.session_state.recorded_file)  # Clean up if needed

    with tab2:
        # Original Upload Functionality
        uploaded_file = st.file_uploader("Or upload a WAV file", type=["wav"])
        if uploaded_file:
            st.audio(uploaded_file)
            if st.button("Transcribe Upload"):
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                waveform = preprocess_audio("temp.wav")
                try:
                    transcription = predict_transcription(waveform)
                    st.subheader("Transcription:")
                    st.success(transcription)

                    # Translate the transcription to Hindi
                    translated_text = translate_text(transcription)
                    st.subheader("Translation to Hindi:")
                    st.success(translated_text)

                except Exception as e:
                    st.error(f"Error during transcription: {e}")
                os.remove("temp.wav")

if __name__ == "__main__":
    main()
