import streamlit as st
import gdown
import os
import shutil
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Function to download and unzip model
def download_model(model_url, zip_filename):
    if not os.path.exists(zip_filename):
        #st.write("Downloading model...")
        gdown.download(model_url, zip_filename, quiet=False)
        #st.write("Download complete.")
        
        #st.write("Unzipping model...")
        shutil.unpack_archive(zip_filename, "chinua-gpt2")
        #st.write("Unzip complete.")
        #st.write("Model available to use")

# Define model URL and file names
model_url = "https://drive.google.com/uc?id=1-Ndi-ycSXouwlspH7zagfbGpTA2oaRLb"  # Corrected to direct download link
zip_filename = "chinua-gpt2.zip"

# Download and unzip the model
download_model(model_url, zip_filename)

# Check if the model directory exists
model_name = "chinua-gpt2"
if not os.path.exists(model_name):
    #st.write(f"Model directory {model_name} does not exist. Please check the download and unzip process.")
#else:
 #   st.write(f"Model directory {model_name} exists. Loading model...")

  #  # Print contents of the model directory for debugging
    st.write("Contents of model directory:")
  #  st.write(os.listdir(model_name))

    # Load the model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    # Streamlit app layout
    st.title("GPT-2 Model for Chinua Achebe: THINGS FALL APART")

    # Initialize conversation history if not already in session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display conversation history
    for entry in st.session_state.history:
        st.write(f"**Q:** {entry['question']}")
        st.write(f"**A:** {entry['answer']}")

    # Input prompt
    prompt = st.text_input("Enter your prompt:")

    if prompt:
        # Generate response
        generated_text = text_generator(
            prompt,
            max_length=100,  # Adjust as needed
            min_length=10,  # Adjust as needed
            num_return_sequences=1,
            temperature=0.7,  
            num_beams=5,

            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=2
        )[0]['generated_text']
        
        # Update conversation history
        st.session_state.history.append({'question': prompt, 'answer': generated_text})
        
        # Display updated conversation history
        st.write("**Q:**", prompt)
        st.write("**A:**", generated_text)
    else:
        st.error("Please enter a prompt to generate a response.")
