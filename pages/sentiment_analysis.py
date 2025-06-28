import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Page setup
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🧠 Sentiment Analysis App")
st.write("Analyze your text's **polarity**, **subjectivity**, and generate a **word cloud**.")

# Step 1: Input Method
input_method = st.radio("Choose input method:", ["Type text", "Upload .txt file"])

user_input = ""

if input_method == "Type text":
    user_input = st.text_area("Enter your text here:")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        user_input = uploaded_file.read().decode("utf-8")

# Step 2: Process Input
if user_input:
    st.subheader("🔍 Analysis Results")

    blob = TextBlob(user_input)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)

    # Label sentiment
    if polarity > 0:
        sentiment_label = "Positive 😊"
    elif polarity < 0:
        sentiment_label = "Negative 😞"
    else:
        sentiment_label = "Neutral 😐"

    st.write(f"**Polarity:** {polarity}")
    st.write(f"**Subjectivity:** {subjectivity}")
    st.write(f"**Sentiment:** {sentiment_label}")

    # Step 3: WordCloud
    st.subheader("☁️ Word Cloud")
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate(user_input)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # Step 4: Optional Sentence-wise Polarity
    st.subheader("📊 Sentence-wise Sentiment")
    sentences = blob.sentences
    data = {
        "Sentence": [str(s) for s in sentences],
        "Polarity": [round(s.sentiment.polarity, 2) for s in sentences]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

    st.line_chart(df.set_index("Sentence"))

