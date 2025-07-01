import streamlit as st
import joblib

model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Spam Email Classifier")
st.write("Enter the email content below to check if it's **SPAM** or **HAM**.")

user_input = st.text_area("Enter Email Content: ")

if st.button("Classify"):
    if user_input.strip() == "":
            st.warning("Please enter a valid email content to classify.")
    else:
        input_vector = vectorizer.transform([user_input])
        
        prediction = model.predict(input_vector)[0]
        prediction_prob = model.predict_proba(input_vector).max()
        
        if prediction == "spam":
            st.error(f"The email is classified as **SPAM** ({prediction_prob: .2%} condidence).")
        else:
            st.success(f"The email is classified as **HAM** ({prediction_prob: .2%} condidence).")