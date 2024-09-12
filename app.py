import streamlit as st
import gdown
import os
import shutil
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
def download_model(model_url, zip_filename):
    if not os.path.exists(zip_filename):
        gdown.download(model_url, zip_filename, quiet=False)
        shutil.unpack_archive(zip_filename, "cmodel-gpt")
model_url = "https://drive.google.com/uc?id=1d-PQHghZxwdNp49ECxCJ917MUMiKdgz7"
zip_filename = "cmodel-gpt.zip"
download_model(model_url, zip_filename)
model_name = "cmodel-gpt"
if not os.path.exists(model_name):
    st.write(f"Model directory {model_name} does not exist. Please check the download and unzip process.")
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    st.title("GPT-2 Model for Chinua Achebe: THINGS FALL APART")

    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'welcomed' not in st.session_state:
        st.session_state.welcomed = False

    if not st.session_state.welcomed:
        welcome_prompt = st.text_input("Say 'hello' to start the conversation:", key="welcome_prompt")
        if welcome_prompt.lower() == 'hello':
            welcome_response = "Welcome! I am here to discuss 'Things Fall Apart' by Chinua Achebe with you. How can I assist you today?"
            st.session_state.history.append({'question': welcome_prompt, 'answer': welcome_response})
            st.write(f"**User😍:** {welcome_prompt}")
            st.write(f"**Chinua's bot😎:** {welcome_response}")
            st.session_state.welcomed = True
    else:
        for entry in st.session_state.history:
            st.write(f"**User😍:** {entry['question']}")
            st.write(f"**Chinua's bot😎:** {entry['answer']}")

        prompt = st.text_input("Enter your prompt:", key="conversation_prompt")

        if prompt:
            result = text_generation_pipeline(prompt, 
                max_length=150,
                min_length=30,
                temperature=4.,
                num_beams=10,
                top_p = 0.7,
                repetition_penalty = 1.5,
                early_stopping = True,
                do_sample = False,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=2,
                truncation=True
            )[0]['generated_text']

            # Process the result to stop at sentence-ending punctuation
            if '.' in result:
                result = result.split('. ')[0] + '.'
            elif '!' in result:
                result = result.split('! ')[0] + '!'

            # Ensure the result does not repeat the question
            answer = result.strip()
            if answer.startswith(prompt):
                answer = answer[len(prompt):].strip()

            st.session_state.history.append({'question': prompt, 'answer': answer})
            st.write("**User😍:**", prompt)
            st.write("**Chinua's bot😎:**", answer)
