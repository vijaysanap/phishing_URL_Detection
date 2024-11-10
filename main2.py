import streamlit as st
import numpy as np
import pickle
from feature import FeatureExtraction  # Ensure this path matches your project structure

# Title of the Streamlit app
st.title("Phishing URL Detection")

# Load the trained model
try:
    # Ensure the model file exists in the correct path and is loaded
    with open("pickle/model.pkl", "rb") as file:  # Adjust the path if necessary
        model = pickle.load(file)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Take URL input from the user
url = st.text_input("Enter URL to check:")

if url:
    try:
        # Feature extraction: Assuming FeatureExtraction takes in a URL and provides a feature list
        obj = FeatureExtraction(url)  # This class should be implemented in your 'feature.py' file
        st.write("Features extracted successfully!")

        # Ensure the number of features expected by the model is 30
        features = np.array(obj.getFeaturesList()).reshape(1, 30)  # 30 features expected by model
        
        # Make prediction with the loaded model
        prediction = model.predict(features)[0]  # Predict whether the URL is phishing (0/1)
        phishing_prob = model.predict_proba(features)[0, 0]  # Probability of phishing
        safe_prob = model.predict_proba(features)[0, 1]  # Probability of safe URL

        # Display the result based on the phishing probability
        if phishing_prob > 0.75:
            # Unsafe URL (phishing probability above 75%)
            st.markdown(
                f"<h1 style='color: red;'>Unsafe URL - {phishing_prob * 100:.2f}% phishing probability</h1>",
                unsafe_allow_html=True
            )
        elif phishing_prob > 0.5:
            # Potentially unsafe URL (phishing probability between 50% and 75%)
            st.markdown(
                f"<h1 style='color: orange;'>Caution - {phishing_prob * 100:.2f}% phishing probability</h1>",
                unsafe_allow_html=True
            )
        else:
            # Safe URL (phishing probability below 50%)
            st.markdown(
                f"<h1 style='color: green;'>Safe URL - {safe_prob * 100:.2f}% non-phishing probability</h1>",
                unsafe_allow_html=True
            )
        
        # Optionally, show the detailed probabilities
        st.write(f"Phishing Probability: {phishing_prob * 100:.2f}%")
        st.write(f"Non-Phishing Probability: {safe_prob * 100:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.write("Please enter a URL to check.")
