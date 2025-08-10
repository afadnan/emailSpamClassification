import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.data.path.append('./nltk_data')


ps = PorterStemmer()

def transform_text(text):
    """
    Performs text preprocessing.
    """
    tokens = nltk.word_tokenize(text.lower())
    
    tokens = [word for word in tokens if word.isalnum()]
    
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    
    tokens = [ps.stem(word) for word in tokens]
    
    return " ".join(tokens)

# Load the saved model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are present.")
    st.stop()


# --- Streamlit App Interface ---

st.title("📧 Email & SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not.")

# Text area for user input
input_sms = st.text_area("Enter the message here...")

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Display
        if result == 1:
            st.header("This looks like Spam", divider='red')
            st.warning("Be cautious with any links or requests in this message.")
        else:
            st.header("This looks like Not Spam", divider='green')
            st.success("This message seems safe.")
    else:
        st.warning("Please enter a message to classify.")