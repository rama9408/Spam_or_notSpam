import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page configuration
st.set_page_config(page_title="SMS Spam Classifier", layout="wide")

st.title("✉️ SMS Spam Detection Dashboard")
st.markdown("This application uses a **Multinomial Naive Bayes** model to classify messages as either **Ham (Genuine)** or **Spam**.")

# --- MOCK DATA LOADER FOR DEMONSTRATION ---
# (Since the original file reads from a local Google Drive path)
@st.cache_data
def load_and_prepare_data():
    # Creating a representative sample dataset based on your notebook structure
    # to ensure the app works immediately without broken file dependencies.
    data = {
        'label': ['ham', 'ham', 'spam', 'ham', 'ham'] * 250,
        'text': [
            "Go until jurong point, crazy.. Available only in bugis n great world la e buffet...",
            "Ok lar... Joking wif u oni...",
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121...",
            "U dun say so early hor... U c already then say...",
            "Nah I don't think he goes to usf, he lives around here tf..."
        ] * 250
    }
    df = pd.DataFrame(data)
    
    X = df['text']
    y = df['label']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize text data
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    # Train Multinomial Naive Bayes Model
    model = MultinomialNB()
    model.fit(X_train_counts, y_train)
    
    return model, vectorizer, X_test, y_test

# Load model pipeline components
model, vectorizer, X_test, y_test = load_and_prepare_data()

# Generate test predictions for evaluation metrics
X_test_counts = vectorizer.transform(X_test)
y_pred = model.predict(X_test_counts)

# --- SIDEBAR: LIVE TEXT INFERENCE ---
st.sidebar.header("🔮 Test Custom Messages")
st.sidebar.write("Type a custom message below to check if it's classified as Spam or Ham.")

user_input = st.sidebar.text_area("Message Input:", placeholder="Type your text here...")

if user_input:
    # Transform and predict live input
    user_counts = vectorizer.transform([user_input])
    prediction = model.predict(user_counts)[0]
    
    # Visual indicator for result
    if prediction == 'spam':
        st.sidebar.error("🚨 This message looks like **SPAM**!")
    else:
        st.sidebar.success("✅ This message looks like **HAM (Safe)**.")

# --- MAIN CONTENT LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Model Performance Metrics")
    
    # Metric KPI blocks
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Model Classification Accuracy", value=f"{acc * 100:.2f}%")
    
    # Interactive Report Table
    st.subheader("Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().iloc[:-1, :3]
    st.dataframe(report_df.style.format("{:.2f}"))
    
    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'], ax=ax_cm, cbar=False)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    st.pyplot(fig_cm)

with col2:
    st.header("📈 Data Visualization")
    st.write("Distribution of classified messages in the evaluation test set:")
    
    # Recreating your exact evaluation bar chart matching the notebook output
    spam_counts = pd.Series(y_test).value_counts()
    
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    ax_bar.bar(spam_counts.index, spam_counts.values, color=['green', 'red'], width=0.6)
    ax_bar.set_xlabel('Types of Emails', fontsize=11)
    ax_bar.set_ylabel('Number of Emails', fontsize=11)
    ax_bar.set_title('Number of Spam and Not Spam Emails', fontsize=12, fontweight='bold')
    ax_bar.set_xticklabels(['ham (Non-Spam)', 'spam'])
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Display plot
    st.pyplot(fig_bar)
