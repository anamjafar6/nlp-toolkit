import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# ---------------------------
# Summarization Function
# ---------------------------
def summarize_text(text, num_sentences, method):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    if method == "LexRank":
        summarizer = LexRankSummarizer()
    elif method == "Luhn":
        summarizer = LuhnSummarizer()
    elif method == "LSA":
        summarizer = LsaSummarizer()
    elif method == "LSA (with stopwords)":
        summarizer = LsaSummarizer(Stemmer("english"))
        summarizer.stop_words = get_stop_words("english")
    else:
        return "Invalid summarization method selected."

    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="📝 Sumy Text Summarizer", layout="centered")
st.title("📝Text Summarizer")

input_method = st.radio("Choose input method:", ["Type Text", "Upload .txt File"])
text = ""

if input_method == "Type Text":
    text = st.text_area("Type your text here:")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

if text:
    st.subheader("📊 Text Stats")
    st.write(f"**Original word count:** {len(text.split())}")

    num_sentences = st.slider("Select number of sentences for summary:", 1, 10, 2)

    method = st.selectbox("Choose summarization method:", ["LexRank", "Luhn", "LSA", "LSA (with stopwords)"])

    if st.button("🔍 Summarize"):
        summary = summarize_text(text, num_sentences, method)
        
        st.subheader("📄 Summary")
        st.write(summary)

        st.download_button(
            label="📥 Download Summary as .txt",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )
