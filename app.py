import streamlit as st
import gdown
import os
import shutil
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Function to download and unzip model
def download_model(model_url, zip_filename):
    if not os.path.exists(zip_filename):
        gdown.download(model_url, zip_filename, quiet=False)
        shutil.unpack_archive(zip_filename, "chinua-gpt2")

# Define model URL and file names
model_url = "https://drive.google.com/uc?id=1-Ndi-ycSXouwlspH7zagfbGpTA2oaRLb"  
zip_filename = "chinua-gpt2.zip"

# Download and unzip the model
download_model(model_url, zip_filename)

# Load the model and tokenizer
model_name = "chinua-gpt2"
if not os.path.exists(model_name):
    st.write(f"Model directory {model_name} does not exist. Please check the download and unzip process.")
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    st.title("GPT-2 Model for Chinua Achebe: THINGS FALL APART")

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'welcomed' not in st.session_state:
        st.session_state.welcomed = False

    if not st.session_state.welcomed:
        # Welcome prompt input
        welcome_prompt = st.text_input("Say 'hello' to start the conversation:", key="welcome_prompt")
        if welcome_prompt.lower() == 'hello':
            welcome_response = "Welcome! I am here to discuss 'Things Fall Apart' by Chinua Achebe with you. How can I assist you today?"
            st.session_state.history.append({'question': welcome_prompt, 'answer': welcome_response})
            st.session_state.welcomed = True
            #st.write(f"**User😍:** {welcome_prompt}")
            #st.write(f"**Chinua's bot😎:** {welcome_response}")
    else:
        # Display conversation history
        for entry in st.session_state.history:
            st.write(f"**User😍:** {entry['question']}")
            st.write(f"**Chinua's bot😎:** {entry['answer']}")

        # Prompt input for conversation
        prompt = st.text_input("Enter your prompt:", key="conversation_prompt")

        if prompt:
            # Generate response
            generated_text = text_generator(
                prompt,
                max_length=100,
                min_length=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=2,
                temperature=0.7,
                num_beams=5,
            )[0]['generated_text']
            
            # Update conversation history
            st.session_state.history.append({'question': prompt, 'answer': generated_text})
            
            # Display updated conversation history
            st.write("**User😍:**", prompt)
            st.write("**Chinua's bot😎:**", generated_text)
