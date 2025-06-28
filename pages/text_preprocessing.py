# Step 1: Import necessary Libraries
import streamlit as st  # for creating web app
import nltk  # for tokenizing words, sentence, POS tagging
import spacy  # for lemmatization and NER(Named Entity Recognition)
from nltk.tokenize import word_tokenize, sent_tokenize 
import io  # to handle uploaded file as a stream

# Step 2: Load spaCy Model
nlp = spacy.load("en_core_web_sm")

# Step 3: Streamlit App Setup 
st.set_page_config(page_title="Basic NLP App", layout="centered")
st.title("Basic NLP Web App")
st.write("Upload a .txt file to analyze your text using NLP tools")

# Upload the file
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("Original Text")
    st.write(text)

    doc = nlp(text)

    st.subheader("Choose NLP Features")
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

    # Show results
    st.subheader("NLP Results")
    st.text(output)

    # Download button 
    buffer = io.BytesIO()
    buffer.write(output.encode())
    buffer.seek(0)

    st.download_button("Download Results", buffer, file_name="nlp_output.txt", mime="text/plain")
