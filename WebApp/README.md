# BRICS Sentiment Analysis Web App

## Goal 🎯
The goal of this sentiment analysis web application is to understand public perceptions about the inclusion of six new member nations into the BRICS alliance. By analyzing public comments, the app helps in organizing and prioritizing insights, detecting sentiment trends, and ensuring that diverse viewpoints are represented. It streamlines the process of understanding public opinion and provides valuable feedback for stakeholders. 🌍🔍

## Model(s) Used for the Web App 🧮
The model used in this web app is a pre-trained LightGBM classifier, which has been fine-tuned for sentiment analysis. The BERT model is used for encoding the text into embeddings, and the LightGBM model predicts the sentiment with high accuracy.



## How to Run the Web App

### Requirements
Ensure you have the necessary libraries and dependencies installed. You can find the list of required packages in the `requirements.txt` file.

### Installation
1. **Clone the repository:**
   ```bash
   gh repo clone Nishant-2728/Sentiment-Analysis-Project
   cd New-BRICS-members-Sentiment-Analysis-518/WebApp
   ```
2. **Install the Dependencies**
  ```bash
  pip install -r requirements.txt
  ```
3. **Run the Streamlit app**
  ```bash
  streamlit run app.py
  ```
