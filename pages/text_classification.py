# 📦 Required Libraries
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

# 📌 Download stopwords
nltk.download('stopwords')

# 🧹 Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 🗂️ Category Mapping (Raw to User-Friendly)
category_map = {
    'alt.atheism': 'Religion/Atheism',
    'comp.graphics': 'Computer Graphics',
    'comp.os.ms-windows.misc': 'Windows OS',
    'comp.sys.ibm.pc.hardware': 'PC Hardware',
    'comp.sys.mac.hardware': 'Mac Hardware',
    'comp.windows.x': 'Windows X',
    'misc.forsale': 'Marketplace/For Sale',
    'rec.autos': 'Automobiles',
    'rec.motorcycles': 'Motorcycles',
    'rec.sport.baseball': 'Baseball',
    'rec.sport.hockey': 'Hockey',
    'sci.crypt': 'Cryptography',
    'sci.electronics': 'Electronics',
    'sci.med': 'Medical/Health',
    'sci.space': 'Space/Science',
    'soc.religion.christian': 'Christian Religion',
    'talk.politics.guns': 'Politics - Guns',
    'talk.politics.mideast': 'Politics - Middle East',
    'talk.politics.misc': 'General Politics',
    'talk.religion.misc': 'General Religion'
}

# 🌟 Streamlit Page Setup
st.set_page_config(page_title="News Article Classifier", layout="centered")
st.title("📰 News Article Classification App")

# 📥 Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})

# ✨ Preprocess the text column
df['text'] = df['text'].apply(preprocess_text)

# 🛠️ Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# 🎯 Train the model
pipeline.fit(df['text'], df['target'])

# 📝 Text Input from User
st.subheader("🔍 Enter a News Article to Classify")
user_input = st.text_area("Type or paste your news article here")

# 🎛 Predict Button with Confidence Score
if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess_text(user_input)
        prediction = pipeline.predict([clean_text])[0]
        
        # Get prediction probabilities
        probs = pipeline.predict_proba([clean_text])[0]
        confidence = round(probs[prediction] * 100, 2)  # Convert to percentage
        
        category_name = newsgroups.target_names[prediction]
        
        st.success(f"Predicted Category: **{category_name}**")
        st.info(f"Model Confidence: **{confidence}%**")
