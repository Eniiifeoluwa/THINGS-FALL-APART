import streamlit as st
import gdown
import os
import shutil
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from langchain import LLMChain, PromptTemplate
from langchain_huggingface import HuggingFacePipeline

def download_model(model_url, zip_filename):
    if not os.path.exists(zip_filename):
        gdown.download(model_url, zip_filename, quiet=False)
        shutil.unpack_archive(zip_filename, "chinua-gpt")
model_url = "https://drive.google.com/uc?id=1bfBeR4C6kUfBvW6JMTTwsEoF3FEyJeCJ"
zip_filename = "chinua-gpt.zip"

download_model(model_url, zip_filename)

model_name = "chinua-gpt"
if not os.path.exists(model_name):
    st.write(f"Model directory {model_name} does not exist. Please check the download and unzip process.")
else:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a knowledgeable historian providing a detailed explanation. Context: {context}\nQuestion: {question}\nAnswer:"
    )
    
    text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    chain = LLMChain(llm=llm, prompt=prompt_template)

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
            context = "The great wrestler in town"  # Adjust the context as needed
            result = chain.run(context=context, question=prompt)
            
            st.session_state.history.append({'question': prompt, 'answer': result})
            st.write("**User😍:**", prompt)
            st.write("**Chinua's bot😎:**", result)
