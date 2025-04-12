import pandas as pd  # For data handling
from transformers import pipeline  # Pre-trained models from Hugging Face
from sklearn.feature_extraction.text import CountVectorizer  # Text to numeric vectors
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.naive_bayes import MultinomialNB  # Simple classifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # Model evaluation
from sklearn.model_selection import cross_val_score  # For cross-validation
import re  # For text cleaning 
import joblib  # For saving and loading the trained model
import streamlit as st  # For building the Streamlit app
import matplotlib.pyplot as plt  # For plotting graphs

# Load dataset
DATA_SET_PATH = "C:/Users/LENOVO/Desktop/project/Emotion_classify_Data.csv"
df = pd.read_csv(DATA_SET_PATH)
print("Dataset Information:")
print(df.info())
print(df.isnull().sum())
if df.isnull().sum().any():
    df = df.dropna()

# Clean text function
def clean_text(text):
    text = text.lower()  # Convert the text to lowercase
    text = re.sub(r'\d+', ' ', text)  # Numbers -> space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    return text

df['cleaned_text'] = df['Comment'].apply(clean_text)  # Use the 'Comment' column for cleaning

# Convert text to numeric vectors
vectorizer = CountVectorizer(stop_words='english')  # Will remove common English words 
X = vectorizer.fit_transform(df['cleaned_text'])  # ['cleaned_text'] is the column containing the preprocessed text

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['Emotion'], test_size=0.2, random_state=42)  # Use the 'Emotion' column for labels

# Train the classifier
classifier = MultinomialNB(alpha=1.0)  # Initialize and train the classifier
classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test)  # Predicting on the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculating accuracy
st.write(f"### Accuracy: {accuracy * 100:.2f}% 🎯")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# Display classification report for detailed performance metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Cross-validation to check for overfitting
scores = cross_val_score(classifier, X, df['Emotion'], cv=5)  # Perform 5-fold cross-validation
print(f"Cross-validation accuracy: {scores.mean():.4f}")
print(f"Standard deviation: {scores.std():.4f}")

# Save the model for later use
joblib.dump(classifier, 'sentiment_model.pkl')

# Load the model later if needed
try:
    classifier = joblib.load('sentiment_model.pkl')
except FileNotFoundError:
    print("Model file not found. Please train the model first.")

# Load the Hugging Face emotion model
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True, device=-1)

st.title('Emotion Classifier 🧠')  # Title of the Streamlit app 
st.write('Enter a text below and the model will predict the emotion. 😃')

# User input
user_input = st.text_area('Enter Text:', '')  # Text area for user input
st.write("---")  # A horizontal line

# Button to trigger the basic prediction
if st.button('Predict Emotion 🤔'):
    if user_input: 
        # Clean and process the input
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])  # Convert the cleaned text into a numerical representation

        # Show a loading spinner while making the prediction
        with st.spinner('Predicting...'):
            # Predict the emotion
            prediction = classifier.predict(input_vector)

        # Display the result 
        st.write(f'Predicted Emotion: {prediction[0]} 😊')
    else:
        st.write('Please enter some text to predict the emotion. 📝')

st.write("---")  # Another horizontal line

emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True, device=-1)

# Button to trigger detailed prediction using Hugging Face
if st.button('Detailed Prediction 🧐'):
    if user_input:
        # Get detailed emotion probabilities from BERT-based model
        results = emotion_pipeline(user_input)
        sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)

        # Display top 5 emotions with scores
        st.subheader('Detailed Emotion Analysis (Top 5): 🔍')

        # Display sorted results with rounded scores
        for item in sorted_results[:5]:
            st.write(f"{item['label']} : {item['score']:.2f}")
    else:
        st.write('Please enter some text to get detailed emotion analysis. 📊')

# Add some space between sections for better layout
st.write("---")  # Another horizontal line

# Function to predict any text
def predict_any_text(input_text):
    cleaned_input = clean_text(input_text)  # Clean the input
    input_vector = vectorizer.transform([cleaned_input])  # Convert the cleaned text into a numerical representation
    prediction = classifier.predict(input_vector)
    return prediction[0]

# Test prediction function with any text
text = "I feel amazing today!"  
print(predict_any_text(text))  

# Plotting a bar chart for top 5 emotions
if user_input:
    results = emotion_pipeline(user_input)
    sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
    
    # Get top 5 emotions and their probabilities
    labels = [item['label'] for item in sorted_results[:5]]
    scores = [item['score'] for item in sorted_results[:5]]

    # Plot the bar chart
    plt.bar(labels, scores, color='blue')
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Top 5 Predicted Emotions')
    st.pyplot(plt)
    
# Custom button with hover effect using CSS
st.markdown(""" 
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)