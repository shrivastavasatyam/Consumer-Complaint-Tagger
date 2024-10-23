import streamlit as st
from models.classifier import classify_input
from models.model_loader import load_models, load_tokenizer

# Load models and tokenizer
tokenizer = load_tokenizer()
models = load_models()

st.title("CFPB Customer Complaint Classification Helper")

st.markdown("""
This application helps classify customer complaints related to various financial products and issues 
handled by the Consumer Financial Protection Bureau (CFPB). Enter the details of a complaint, 
and the system will predict the relevant product, sub-product, issue, and sub-issue categories.
""")

# Text input
input_text = st.text_area("Enter the customer complaint narrative for classification:")


if st.button("Classify"):
    # Classify the input if it's not empty
    if input_text.strip():
        results = classify_input(input_text, models, tokenizer)

        # Display the results
        for model_name, (label, prob) in results.items():
            st.write(f"**{model_name}:** {label} (Prob: {prob:.2f})")
    else:
        st.write("Please enter some text to classify.")
