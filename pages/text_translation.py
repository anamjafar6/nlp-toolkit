import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import os

# ----------------------------
# Language Mapping for gTTS
# ----------------------------
lang_codes = {
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese (Simplified)": "zh-CN",
    "Hindi": "hi",
    "Arabic": "ar",
    "Russian": "ru",
    "Japanese": "ja",
    "English": "en",
    "Urdu": "ur"
}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🌍 LinguaBridge - Text & Voice Translator")
st.title("🌍 LinguaBridge - Text & Voice Translator")

text_input = st.text_area("Enter text to translate:")

target_lang = st.selectbox("Translate to:", list(lang_codes.keys()))

if st.button("Translate"):
    if not text_input.strip():
        st.warning("Please enter text to translate.")
    else:
        try:
            target_code = lang_codes[target_lang]

            # Translate
            translated_text = GoogleTranslator(source='auto', target=target_code).translate(text_input)
            st.success(f"Translated Text ({target_lang}):")
            st.write(translated_text)

            # Text-to-speech
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts = gTTS(translated_text, lang=target_code)
                tts.save(tmp_file.name)
                st.audio(tmp_file.name, format="audio/mp3")

                with open(tmp_file.name, "rb") as f:
                    st.download_button(
                        label="📥 Download Audio",
                        data=f,
                        file_name="translated_audio.mp3",
                        mime="audio/mp3"
                    )

            os.remove(tmp_file.name)

        except Exception as e:
            st.error(f"An error occurred: {e}")
