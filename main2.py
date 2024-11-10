# import streamlit as st
# import numpy as np
# import pickle
# from feature import FeatureExtraction  # Ensure that this is the correct import path

# # Title of the Streamlit app
# st.title("URL Safety Checker")

# # Load the trained model
# try:
#     file = open("pickle/model.pkl", "rb")  # Ensure the model file exists in the correct path
#     gbc = pickle.load(file)
#     file.close()
#     st.write("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model: {e}")

# # Take URL input from the user
# url = st.text_input("Enter URL to check:")

# if url:
#     try:
#         # Feature extraction
#         obj = FeatureExtraction(url)
#         st.write("Features extracted successfully!")
#         x = np.array(obj.getFeaturesList()).reshape(1, 30)  # Ensure that the number of features is 30

#         # Make prediction with the loaded model
#         y_pred = gbc.predict(x)[0]
#         y_pro_phishing = gbc.predict_proba(x)[0, 0]
#         y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

#         # # Determine whether the URL is safe or unsafe based on the prediction
#         # if y_pred == 1:
#         #     pred = f"It is {y_pro_phishing * 100:.2f}% safe to go."
#         # else:
#         #     pred = f"It is {y_pro_non_phishing * 100:.2f}% unsafe to go."
#                 # Display the result based on the phishing probability
#         if y_pro_phishing > 0.75:
#             # Unsafe URL (phishing probability above 75%)
#             st.markdown(
#                 f"<h1 style='color: red;'>Unsafe URL - {y_pro_phishing * 100:.2f}% phishing probability</h1>",
#                 unsafe_allow_html=True
#             )
#         else:
#             # Safe URL (phishing probability below 25%)
#             st.markdown(
#                 f"<h1 style='color: green;'>Safe URL - {y_pro_non_phishing * 100:.2f}% non-phishing probability</h1>",
#                 unsafe_allow_html=True
#             )

#         # Display the prediction and probabilities
#         # st.write(pred)
#         st.write(f"Phishing Probability: {y_pro_phishing * 100:.2f}%")
#         st.write(f"Non-Phishing Probability: {y_pro_non_phishing * 100:.2f}%")

#     except Exception as e:
#         st.error(f"Error during prediction: {e}")
# else:
#     st.write("Please enter a URL to check.")

import streamlit as st
import numpy as np
import pickle
from feature import FeatureExtraction  # Ensure that this is the correct import path

# Title of the Streamlit app
st.title("URL Safety Checker")

# Load the trained model
try:
    file = open("pickle/model.pkl", "rb")  # Ensure the model file exists in the correct path
    gbc = pickle.load(file)
    file.close()
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Take URL input from the user
url = st.text_input("Enter URL to check:")

if url:
    try:
        # Feature extraction
        obj = FeatureExtraction(url)
        st.write("Features extracted successfully!")
        x = np.array(obj.getFeaturesList()).reshape(1, 30)  # Ensure that the number of features is 30

        # Make prediction with the loaded model
        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

        # Display the result based on the phishing probability
        if y_pro_phishing > 0.75:
            # Unsafe URL (phishing probability above 75%)
            st.markdown(
                f"<h1 style='color: red;'>Unsafe URL - {y_pro_phishing * 100:.2f}% phishing probability</h1>",
                unsafe_allow_html=True
            )
        else:
            # Safe URL (phishing probability below 25%)
            st.markdown(
                f"<h1 style='color: green;'>Safe URL - {y_pro_non_phishing * 100:.2f}% non-phishing probability</h1>",
                unsafe_allow_html=True
            )
        
        # Optionally, show the detailed probabilities
        st.write(f"Phishing Probability: {y_pro_phishing * 100:.2f}%")
        st.write(f"Non-Phishing Probability: {y_pro_non_phishing * 100:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.write("Please enter a URL to check.")
