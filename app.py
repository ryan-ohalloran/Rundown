#!/usr/bin/env python3

import streamlit as st
from api import Client

# retrieve API key from streamlit secret
OPENAI_KEY = st.secrets["OPENAI_KEY"]

# instantiate client
client = Client(OPENAI_KEY)

st.title("Rundown")
st.write("A tool for summarizing audio recordings.")

st.header("Upload audio file")
audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "mp4", "m4a", "mpg", "mpga", "webm"])

# transcribe audio file
st.write("Transcribe")
transcript = ""
if audio_file is not None:
    transcript = client.transcribe(audio_file)
    st.write("Transcript: " + transcript)

# get prompt instructions as list of text inputs
st.header("Prompt instructions")

# Use a session state to store the number of text boxes
if 'num_boxes' not in st.session_state:
    st.session_state['num_boxes'] = 1

# Create the text boxes
prompt_instructions = []
for i in range(st.session_state['num_boxes']):
    prompt_instructions.append(st.text_input(f"Prompt instruction {i+1}", key=f"Prompt instruction {i+1}"))

# Button to add more text boxes
if st.button('Add another instruction'):
    st.session_state['num_boxes'] += 1

# specify model as a text entry
st.header("Model")
model = st.text_input("Model", value="gpt-4")

# specify temperature as a slider
st.header("Temperature")
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.9, step=0.01)

# summarize audio file
st.header("Summarize audio file")
summary = client.summarize(transcript, prompt_instructions, model=model, temperature=temperature)
st.write("Summary: " + summary)
