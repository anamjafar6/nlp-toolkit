import streamlit as st

# ----------------- Page Config -----------------
st.set_page_config(page_title="NLP Toolkit by Anam", layout="centered")

# ----------------- Apply Custom CSS -----------------
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        font-size:36px;
        color:#4b4bff;
    }
    .footer {
        text-align:center;
        color:gray;
        font-size:12px;
        margin-top:50px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Sidebar with Logo -----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/9133/9133080.png", width=100)  # Replace with your logo URL or file
st.sidebar.title("NLP Toolkit Menu")

# ----------------- Main Page -----------------
st.markdown('<div class="title">💡 NLP Toolkit by Anam</div>', unsafe_allow_html=True)

st.write("""
Welcome to your multi-functional **NLP web app**!  
Explore various NLP features like:

- 🔧 Text Cleaning  
- 📝 Summarization  
- 😊 Sentiment Analysis  
- 🌍 Text & Voice Translation  
- 📰 News Article Classification  

👉 Use the **sidebar** to navigate different features.  
""")

# ----------------- Footer -----------------
st.markdown('<div class="footer">Developed with ❤️ by Anam | Powered by Streamlit</div>', unsafe_allow_html=True)
