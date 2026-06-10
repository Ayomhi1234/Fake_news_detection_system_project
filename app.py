import streamlit as st
import joblib

st.title("Nigerian Fake News Detector")
st.write("Paste a news article to check if it's fake or real")

# Load model and vectorizer
model = joblib.load("News_Project.joblib")
vectorizer = joblib.load("News_tfidf_vectorizer.joblib")

# Input
user_text = st.text_area("Paste article here:")

if st.button("Predict"):
    if user_text:
        # Vectorize
        X = vectorizer.transform([user_text])
        
        # Predict
        prediction = model.predict(X)[0]
        confidence = model.decision_function(X)[0]
        
        # Display
        if prediction == 1:
            st.error(f"🚨 FAKE NEWS (Confidence: {abs(confidence):.2f})")
        else:
            st.success(f"✅ REAL NEWS (Confidence: {abs(confidence):.2f})")
    else:
        st.warning("Please paste an article first")