import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Point NLTK to the local 'nltk_data' directory
# This line is crucial for deployment on platforms like Render or Streamlit Cloud
nltk.data.path.append('./nltk_data')

# Initialize the Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    """
    Performs text preprocessing:
    1. Converts to lowercase
    2. Tokenizes into words
    3. Removes non-alphanumeric characters
    4. Removes stopwords and punctuation
    5. Applies stemming
    """
    # Lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # Keep only alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    
    # Apply stemming
    tokens = [ps.stem(word) for word in tokens]
    
    return " ".join(tokens)

# Load the saved TF-IDF vectorizer and model
# Ensure these .pkl files are in the same directory as app.py
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the project directory.")
    st.stop()


# --- Streamlit App Interface ---

st.title("📧 Email & SMS Spam Classifier")
st.write("Enter a message below to check if it's spam or not.")

# Text area for user input
input_sms = st.text_area("Enter the message here...")

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict using the loaded model
        result = model.predict(vector_input)[0]
        
        # 4. Display the result
        if result == 1:
            st.header("This looks like Spam", divider='red')
            st.warning("Be cautious with any links or requests in this message.")
        else:
            st.header("This looks like Not Spam", divider='green')
            st.success("This message seems safe.")
    else:
        st.warning("Please enter a message to classify.")