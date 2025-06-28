# ----------------- IMPORTS -----------------

import streamlit as st  # to build the web app interface
import nltk  # for tokenizing text and stopwords
import spacy  # advanced NLP tasks like lemmatization, POS tagging, etc.
import io  # to handle file uploads/downloads in memory
import re  # for regular expressions - text cleaning
import pandas as pd  # to handle tabular data (dataframes)
from nltk.corpus import stopwords  # list of common stopwords in English
from textblob import TextBlob  # for sentiment analysis
from wordcloud import WordCloud  # to generate word cloud images
from sklearn.feature_extraction.text import TfidfVectorizer  # to convert text into numerical features
from sklearn.linear_model import LogisticRegression  # machine learning model for classification
from sklearn.pipeline import Pipeline  # to combine preprocessing + model steps
from sklearn.datasets import fetch_20newsgroups  # built-in dataset for news classification
from sumy.parsers.plaintext import PlaintextParser  # parses raw text for summarization
from sumy.nlp.tokenizers import Tokenizer  # tokenizes text into sentences
from sumy.summarizers.lex_rank import LexRankSummarizer  # LexRank summarizer algorithm
from sumy.summarizers.luhn import LuhnSummarizer  # Luhn summarizer algorithm
from sumy.summarizers.lsa import LsaSummarizer  # LSA summarizer algorithm
from sumy.nlp.stemmers import Stemmer  # for stemming in summarizer
from sumy.utils import get_stop_words  # get stopwords for summarizer
from deep_translator import GoogleTranslator  # for translating text into different languages
from gtts import gTTS  # Google Text-to-Speech to convert text to audio
import tempfile  # to create temporary files for audio
import os  # to handle file system tasks (deleting temporary files)
import speech_recognition as sr  # speech to text conversion
from pydub import AudioSegment  # to convert audio files (e.g., mp3 to wav)

# ----------------- SETUP -----------------

nltk.download('stopwords')  # download stopwords for cleaning
spacy_model = spacy.load("en_core_web_sm")  # load English model for spaCy (NLP tasks)

# Streamlit Page Configurations with better layout and wide mode for good look
st.set_page_config(page_title="NLP Toolkit by Anam", layout="wide")

# Add some custom styling for fonts and headings
st.markdown("""
    <style>
        .big-font {
            font-size:35px !important;
            font-weight:bold;
            color:#4B8BBE;
        }
        .sub-font {
            font-size:18px !important;
            color:#333333;
        }
    </style>
""", unsafe_allow_html=True)

# Main Heading with bigger font
st.markdown("<div class='big-font'>💡 Welcome to NLP Toolkit by Anam</div>", unsafe_allow_html=True)

# Subtext for app description
st.markdown("<div class='sub-font'>Unlock the power of Natural Language Processing with this interactive toolkit. Explore features using the sidebar!</div>", unsafe_allow_html=True)

# Sidebar Options
option = st.sidebar.radio("Select a feature:", [
    "Text Processing", 
    "Sentiment Analysis", 
    "News Classification",
    "Text Cleaning",
    "Text Summarizer",
    "Text Translation",
    "Voice Translation"
])

# ----------------- TEXT PROCESSING -----------------

if option == "Text Processing":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])  # user uploads text file
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")  # decode uploaded file
        st.subheader("Original Text")
        st.write(text)

        doc = spacy_model(text)  # process text using spaCy

        # checkboxes for different NLP tasks
        sent_tok = st.checkbox("Sentence Tokenization")
        word_tok = st.checkbox("Word Tokenization")
        pos_tag_check = st.checkbox("Part of Speech Tags")
        ner_tag_check = st.checkbox("Named Entity Recognition")
        lemma_tag = st.checkbox("Lemmatization")

        output = ""

        if sent_tok:
            output += "Sentences:\n" + "\n".join([sent.text for sent in doc.sents]) + "\n\n"
        if word_tok:
            output += "Word Tokens:\n" + ", ".join([token.text for token in doc]) + "\n\n"
        if pos_tag_check:
            output += "Part of Speech Tags:\n" + "\n".join([f"{token.text} → {token.pos_}" for token in doc]) + "\n\n"
        if ner_tag_check:
            output += "Named Entities:\n" + "\n".join([f"{ent.text} → {ent.label_}" for ent in doc.ents]) + "\n\n"
        if lemma_tag:
            output += "Lemmas:\n" + "\n".join([f"{token.text} → {token.lemma_}" for token in doc]) + "\n\n"

        st.subheader("NLP Results")
        st.text(output)

        # allow user to download results
        buffer = io.BytesIO()
        buffer.write(output.encode())
        buffer.seek(0)
        st.download_button("Download Results", buffer, file_name="nlp_output.txt", mime="text/plain")

# ----------------- SENTIMENT ANALYSIS -----------------

elif option == "Sentiment Analysis":
    input_method = st.radio("Choose input method:", ["Type text", "Upload .txt file"])
    user_input = ""

    if input_method == "Type text":
        user_input = st.text_area("Enter your text here:")
    else:
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file:
            user_input = uploaded_file.read().decode("utf-8")

    if user_input:
        blob = TextBlob(user_input)  # analyze text
        polarity = round(blob.sentiment.polarity, 2)  # sentiment polarity: -1 to +1
        subjectivity = round(blob.sentiment.subjectivity, 2)  # subjectivity score

        # label based on polarity
        sentiment_label = "Positive 😊" if polarity > 0 else "Negative 😞" if polarity < 0 else "Neutral 😐"

        st.write(f"**Polarity:** {polarity}")
        st.write(f"**Subjectivity:** {subjectivity}")
        st.write(f"**Sentiment:** {sentiment_label}")

        st.subheader("☁️ Word Cloud")
        wordcloud = WordCloud(width=600, height=300, background_color='white').generate(user_input)
        st.image(wordcloud.to_array())  # display wordcloud

# ----------------- NEWS CLASSIFICATION -----------------

elif option == "News Classification":
    newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)  # load dataset
    df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})  # create dataframe

    # basic cleaning function
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    df['text'] = df['text'].apply(preprocess_text)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(df['text'], df['target'])  # train model

    user_input = st.text_area("Enter a news article to classify:")
    if st.button("Predict Category") and user_input.strip():
        clean_text = preprocess_text(user_input)
        prediction = pipeline.predict([clean_text])[0]
        probs = pipeline.predict_proba([clean_text])[0]
        confidence = round(probs[prediction] * 100, 2)
        category_name = newsgroups.target_names[prediction]
        st.success(f"Predicted Category: {category_name}")
        st.info(f"Model Confidence: {confidence}%")

# ----------------- TEXT CLEANING -----------------

elif option == "Text Cleaning":
    uploaded_file = st.file_uploader("Upload your text file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.write(text)

        to_lower = st.checkbox("Convert to lowercase")
        remove_punc = st.checkbox("Remove Punctuations")
        remove_nums = st.checkbox("Remove numbers")
        remove_sw = st.checkbox("Remove stopwords")

        def clean_my_text(text):
            if to_lower:
                text = text.lower()
            if remove_punc:
                text = re.sub(r'[^\w\s]', '', text)
            if remove_nums:
                text = re.sub(r'\d+', '', text)
            if remove_sw:
                stop_words = set(stopwords.words("english"))
                words = text.split()
                words = [word for word in words if word not in stop_words]
                text = ' '.join(words)
            return text

        cleaned = clean_my_text(text)
        st.write(cleaned)

# ----------------- TEXT SUMMARIZER -----------------

elif option == "Text Summarizer":
    input_method = st.radio("Choose input method:", ["Type Text", "Upload .txt File"])
    text = ""

    if input_method == "Type Text":
        text = st.text_area("Type your text here:")
    else:
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")

    if text:
        num_sentences = st.slider("Select number of sentences for summary:", 1, 10, 2)
        method = st.selectbox("Choose summarization method:", ["LexRank", "Luhn", "LSA", "LSA (with stopwords)"])

        if st.button("Summarize"):
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = None
            if method == "LexRank":
                summarizer = LexRankSummarizer()
            elif method == "Luhn":
                summarizer = LuhnSummarizer()
            elif method == "LSA":
                summarizer = LsaSummarizer()
            elif method == "LSA (with stopwords)":
                summarizer = LsaSummarizer(Stemmer("english"))
                summarizer.stop_words = get_stop_words("english")

            summary = summarizer(parser.document, num_sentences)
            st.write(" ".join(str(sentence) for sentence in summary))

# ----------------- TEXT TRANSLATION -----------------

elif option == "Text Translation":
    text_input = st.text_area("Enter text to translate:")
    lang_codes = {"French": "fr", "Spanish": "es", "German": "de", "Hindi": "hi", "Urdu": "ur"}
    target_lang = st.selectbox("Translate to:", list(lang_codes.keys()))

    if st.button("Translate") and text_input.strip():
        translated_text = GoogleTranslator(source='auto', target=lang_codes[target_lang]).translate(text_input)
        st.success(f"Translated Text ({target_lang}):")
        st.write(translated_text)

# ----------------- VOICE TRANSLATION -----------------

elif option == "Voice Translation":
    lang_codes = {'english': 'en', 'french': 'fr', 'german': 'de', 'spanish': 'es', 'urdu': 'ur'}
    target_lang_name = st.selectbox("Choose language to translate to:", list(lang_codes.keys()))
    target_lang_code = lang_codes[target_lang_name]

    audio_file = st.file_uploader("Upload a voice file (WAV/MP3)", type=["wav", "mp3"])
    if audio_file:
        temp_audio_path = "temp_audio.wav"

        if audio_file.name.endswith(".mp3"):
            audio = AudioSegment.from_mp3(audio_file)
            audio.export(temp_audio_path, format="wav")
        else:
            with open(temp_audio_path, "wb") as f:
                f.write(audio_file.read())

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            try:
                original_text = recognizer.recognize_google(audio_data)
                st.write("Detected Text:", original_text)

                translated_text = GoogleTranslator(source='auto', target=target_lang_code).translate(original_text)
                st.write("Translated Text:", translated_text)

                tts = gTTS(text=translated_text, lang=target_lang_code)
                tts.save("translated.mp3")
                st.audio("translated.mp3", format="audio/mp3")

            except Exception as e:
                st.error(f"Error: {e}")

        os.remove(temp_audio_path)
        if os.path.exists("translated.mp3"):
            os.remove("translated.mp3")
