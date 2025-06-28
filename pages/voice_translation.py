import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import io
import os
from pydub import AudioSegment

# -----------------------
# Helper: Map language names to gTTS codes
# -----------------------
lang_codes = {
    'english': 'en',
    'french': 'fr',
    'german': 'de',
    'spanish': 'es',
    'italian': 'it',
    'chinese (simplified)': 'zh-CN',
    'japanese': 'ja',
    'hindi': 'hi',
    'urdu': 'ur',
    'arabic': 'ar',
    'russian': 'ru',
    'portuguese': 'pt',
    # Add more as needed
}

# -----------------------
# Streamlit Page Setup
# -----------------------
st.set_page_config(page_title="🎤 Voice Translator", layout="centered")
st.title("🎤 Voice Translation App")
st.write("Upload or record your voice, and translate it into another language!")

# -----------------------
# Language Selection
# -----------------------
languages = list(lang_codes.keys())
target_lang_name = st.selectbox("🌍 Choose language to translate to:", languages)
target_lang_code = lang_codes[target_lang_name]

# -----------------------
# Upload audio file
# -----------------------
audio_file = st.file_uploader("📁 Upload a voice file (WAV/MP3)", type=["wav", "mp3"])

# -----------------------
# Voice Translation Process
# -----------------------
if audio_file is not None:
    temp_audio_path = "temp_audio.wav"

    # Convert MP3 to WAV if needed
    if audio_file.name.endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_file)
        audio.export(temp_audio_path, format="wav")
    else:
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.read())

    recognizer = sr.Recognizer()

    with sr.AudioFile(temp_audio_path) as source:
        st.info("🔎 Converting voice to text...")

        # Adjust for noise & record
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)

        try:
            # You can specify language='en-US' if needed
            original_text = recognizer.recognize_google(audio_data)
            st.subheader("📄 Detected Text")
            st.write(original_text)

            # Translate the text
            translated_text = GoogleTranslator(source='auto', target=target_lang_code).translate(original_text)
            st.subheader("🌐 Translated Text")
            st.write(translated_text)

            # Convert to speech
            tts = gTTS(text=translated_text, lang=target_lang_code)
            tts.save("translated.mp3")

            # Play audio
            st.subheader("🔊 Listen to Translation")
            st.audio("translated.mp3", format="audio/mp3")

            # Download button
            with open("translated.mp3", "rb") as audio_file:
                st.download_button(
                    label="📥 Download Translated Audio",
                    data=audio_file,
                    file_name="translated_audio.mp3",
                    mime="audio/mp3"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Clean up
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    if os.path.exists("translated.mp3"):
        os.remove("translated.mp3")
