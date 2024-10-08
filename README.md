# GPT-2 Fine-Tuned on "Things Fall Apart" by Chinua Achebe

This repository contains a project where I fine-tuned GPT-2 on the text of Chinua Achebe's *Things Fall Apart*. The model is deployed using Streamlit, allowing users to interact with a language model that generates text in the style of the novel.

## Project Overview

### 1. Fine-Tuning GPT-2

I fine-tuned OpenAI's GPT-2 model on the text of *Things Fall Apart* to generate responses and text that mimic the style and themes of the book. This involved:

- **Data Preparation**: Extracting and preprocessing the text from the novel.
- **Model Fine-Tuning**: Using the Hugging Face Transformers library to fine-tune GPT-2 on the prepared text.
- **Model Saving**: Saving the fine-tuned model for later use in inference.

### 2. Streamlit Deployment

The fine-tuned GPT-2 model is deployed using Streamlit, enabling an interactive web interface where users can:

- Start a conversation by saying "hello" and receive a warm welcome.
- Ask questions or input prompts related to *Things Fall Apart* and receive contextually relevant responses generated by the model.
- View a conversation history that tracks the interaction between the user and the model.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Eniiifeoluwa/things-fall-apart.git
   cd things-fall-apart
2. **Install required packages and run the app**:
```bash
pip install -r requirements.txt
streamlit run app.py
