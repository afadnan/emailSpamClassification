# Email Spam Classifier 📧
This project is a machine learning-based application designed to classify text messages (like emails or SMS) as either "Spam" or "Not Spam" (often referred to as "Ham"). It uses a Multinomial Naive Bayes model trained on a dataset of SMS messages. The project includes a Jupyter Notebook detailing the data analysis and model training process, and a user-friendly web interface built with Streamlit for real-time predictions.

## 📋 Project Architecture
The project follows a standard machine learning workflow, from data exploration to deployment. The architecture can be broken down as follows:

User Input (Message)
       |
       v
[ Streamlit Web App (app.py) ]
       |
       v
[ Text Preprocessing Function ]
(Lowercase, Tokenize, Remove special chars, Remove stopwords, Stemming)
       |
       v
[ TF-IDF Vectorization ]
(Loads vectorizer.pkl)
       |
       v
[ Multinomial Naive Bayes Model ]
(Loads model.pkl)
       |
       v
Prediction Output ("Spam" or "Not Spam")

## ⚙️ Project Workflow
The analysis and model development were carried out in the spamClassification.ipynb notebook, following these key steps:

### 1. Data Cleaning
Loaded the Dataset: Started with the SMS Spam Collection dataset from a .csv file.

Removed Redundancy: Dropped irrelevant columns and removed duplicate entries to ensure data quality.

Standardized Labels: Converted the categorical labels ('ham', 'spam') into numerical format (0 for ham, 1 for spam) using a label encoder.

### 2. Exploratory Data Analysis (EDA)
Class Distribution: Analyzed the distribution of spam vs. ham messages, revealing an imbalanced dataset (approximately 87% ham, 13% spam).

Feature Engineering: Created new features to better understand the data, such as:

Number of characters

Number of words

Number of sentences

Visual Analysis: Plotted histograms and generated word clouds for both spam and ham messages. Key insights included:

Spam messages generally contain more characters and words.

Words like "free," "win," "claim," "prize," and "urgent" were highly frequent in spam messages.

### 3. Text Preprocessing
A comprehensive preprocessing pipeline was created to transform raw text data into a clean, machine-readable format. The steps included:

Lowercasing: Converted all text to lowercase.

Tokenization: Broke down sentences into individual words (tokens).

Special Character Removal: Filtered out non-alphanumeric characters.

Stopword & Punctuation Removal: Eliminated common English stopwords (e.g., 'the', 'a', 'in') and punctuation.

Stemming: Reduced words to their root form (e.g., 'dancing' -> 'danc') using the Porter Stemmer.

### 4. Model Building
Vectorization: Used the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert the preprocessed text into numerical feature vectors, limiting the features to the top 3000 most frequent words.

Model Comparison: Trained and evaluated eleven different classification algorithms to find the best performer for this task, including:

Naive Bayes (Gaussian, Multinomial, Bernoulli)

Logistic Regression

Support Vector Machine (SVC)

K-Nearest Neighbors (KNN)

Decision Tree & Random Forest

Ensemble methods like AdaBoost, Bagging, and Gradient Boosting

## 🏆 Model Selection and Performance
After a thorough comparison of all models, the Multinomial Naive Bayes (MNB) classifier was selected. It provided an excellent balance of high accuracy and perfect precision, which is crucial for a spam classifier (minimizing the chance of incorrectly classifying a legitimate message as spam).

The performance of the selected MNB model on the test dataset was:

Metric	Score
Accuracy	97.10%
Precision	1.00

Export to Sheets
Confusion Matrix:

[[896   0]
 [ 30 108]]
True Negatives (Ham correctly identified): 896

False Positives (Ham incorrectly marked as Spam): 0

False Negatives (Spam incorrectly marked as Ham): 30

True Positives (Spam correctly identified): 108

The Precision score of 1.0 is particularly noteworthy, as it means that out of all the messages the model predicted as spam, 100% were actually spam.

## 💻 Technologies & Libraries Used
Language: Python 3

Data Analysis: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn, WordCloud

Text Processing: NLTK (Natural Language Toolkit)

Machine Learning: Scikit-learn, XGBoost

Web App: Streamlit

## 🚀 Setup and Usage
To run this project on your local machine, follow these steps:

Clone the Repository

Bash

git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
Create a Virtual Environment

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Dependencies
Create a requirements.txt file with the following content:

pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
nltk
wordcloud
xgboost
Then, install the libraries:

Bash

pip install -r requirements.txt
Download NLTK Data
Run the following command in a Python interpreter to download the necessary NLTK packages:

Python

import nltk
nltk.download('punkt')
nltk.download('stopwords')
Run the Streamlit App

Bash

streamlit run app.py
Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

## 🤝 How to Contribute
Contributions, issues, and feature requests are welcome! If you have suggestions for improvement, please feel free to open an issue or submit a pull request on GitHub.
