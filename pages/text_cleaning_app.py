# Import required libraries
import streamlit as st # to build a web app using python
import re # it is a tool which will find and remove patterns(punctuation, numbers) in text
import nltk # it helps download stopwords (common useless words like the, is , at)
from nltk.corpus import stopwords # this way we will get a list of stopwords in english
import io # this library handle files in the memory(like saving cleaned text and allowing download)

# Downloading stopwords for the first time
nltk.download('stopwords')

# streamlit page setup
st.set_page_config(page_title = "Simple Text Cleaner", layout = "centered") # st.setpage_config is an streamlit function which is use to set title and layout of the webapp
st.title("🧹Text Cleaning App") # It displays main heading of the webapp
st.write("Upload a .txt file and choose the cleaning opptions below: ") # st.write is use to display normal text or you can give any inforamtion by using this

# Upload file
uploaded_file = st.file_uploader("Upload your text file", type=["txt"]) # from this user can upload his file

# show original text
if uploaded_file is not None: # this will check if uploaded file is not empty
    text = uploaded_file.read().decode("utf-8") # uploaded file reads the data inside the file but in bytes format and decode then convert it into normal string
    st.subheader("📄Original Text") 
    st.write(text)

    # Show cleaning options
    st.subheader("⚙️Choose Cleaning Options") # this will show the user so that they will choose the cleaning options

# we are making checkboxes 
    to_lower = st.checkbox("Convert to lowercase") # this will make  acheckbox named convert to lower case
    remove_punc = st.checkbox("remove Punctuations") #this will make a checkbox named remove punctutation
    remove_nums = st.checkbox("Remove numbers") # this will make a checkbox named remove numbers
    remove_sw = st.checkbox("Remove stopwords (common words)")  # this will make a check box named Remove stopwords
    remove_urls = st.checkbox("Remove URLs")  # Checkbox for URL removal

    # step 6: define a cleaning function
    def clean_my_text(text): # we made this function to clean the text based on the checkboxes the user 
        # remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        if to_lower:
            text = text.lower() # if user select theonvert to lower case checkbox this function will run and it convert it into lower case

        if remove_punc: # re.sub() means replace a text using pattern
            text = re.sub(r'[^\w\s]', '', text) # pattern [^\w\s] means anything that is NOt a number , letter or space so it will remove special characters like .,!?

        if remove_nums: # if user select remove number check box this will remove numbers (87 467) from the text
            text = re.sub(r'\d+', '', text) # \d+ means one or more digit , '' means replace with nothing

        if remove_sw:
            stop_words = set(stopwords.words("english")) # set() makes it fast to search nad get a list of stop words form NLTK
            words = text.split() # it will break text into individual words
            words = [word for word in words if word not in stop_words] # remove all stopwords fro the text
            text = ' '.join(words) # join the words back together into a cleaned sentence

        return text # this willl gives the final cleaned text back so we can show or download it
    

    # step 7 : Clean the text based on selected options
    cleaned = clean_my_text(text)

    # Step 8 : Show cleaned Text
    st.subheader("✅cleaned Text")
    st.write(cleaned)


    # Step 9 : Add download button
    buffer = io.BytesIO() # we are creating an empty box (buffer) whre we'll store the cleaned text temporarily
    buffer.write(cleaned.encode()) # we are writing the cleaned text into that buffer, encode() means chnages the text into bytes(computer-readable format)
    buffer.seek(0) #this moves the "file pointer" to the start of the buffer, so that when we click download, it starts reading from the begining of the file

    st.download_button(
        label = "📥Download Cleaned Text", # this way user will download cleaned text
        data = buffer, # it will read data from the buffer file
        file_name = "cleaned_text.txt", # it is the name of the downloaded file
        mime="text/plain" # this tells the browser is a plain text file (not image, vedio, etc)
    )


