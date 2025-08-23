import streamlit as st
import pickle
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os

# Relative path to Model folder
model_path = os.path.join(os.path.dirname(__file__), "..", "Model", "lgb_model.pkl")

try:
    with open(model_path, 'rb') as file:
        lgb_model = pickle.load(file)
except FileNotFoundError:
    lgb_model = None
    st.error("Model file not found. Please ensure 'lgb_model.pkl' exists in the Model folder.")

# Load the BERT tokenizer and model for encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_text(text):
    encoded_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state[:, 0, :].numpy()

# Function to perform sentiment prediction
def predict_sentiment(text_input):
    if lgb_model is None:
        return "Error: Model not loaded"
    encoded_text = encode_text(text_input)
    with torch.no_grad():
        prediction = lgb_model.predict(encoded_text)
    return 'Positive' if prediction[0] > 0.5 else 'Negative'

# Streamlit app
def main():
    st.title('Sentiment Analysis Web App')
    user_input = st.text_input('Enter the text for sentiment analysis:')
    if st.button('Predict'):
        if user_input:
            sentiment = predict_sentiment(user_input)
            st.write(f'Sentiment: {sentiment}')
        else:
            st.warning('⚠️ Please enter some text to analyze.')

if __name__ == "__main__":
    main()
