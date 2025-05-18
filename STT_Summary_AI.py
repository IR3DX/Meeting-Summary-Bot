import torch
import os
import re
import gradio as gr
from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# LLM Setup

my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}

params = {
        GenParams.MAX_NEW_TOKENS: 700, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. 
    }

LLAMA3_model = Model(
        model_id= 'meta-llama/llama-3-2-11b-vision-instruct', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network",  
        )

llm = WatsonxLLM(LLAMA3_model)  

# Prompt:

temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template= temp)

prompt_to_LLAMA3 = LLMChain(llm=llm, prompt=pt)

# Cleaning Llama output

def clean_llama_output(raw_output):
    # Remove all tags like <s>, <<SYS>>, [INST], etc.
    return re.sub(r"<[^>]*>|\[[^\]]*\]", "", raw_output).strip()

# STT:

def Summarize_audio(audio_file):
    # Initialize the speech recognition pipeline
    
    pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
)   
    # Transcribe the audio file and return the result
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    # run the chain to merge transcript text with the template and send it to the LLM
    raw_result = prompt_to_LLAMA3.run(transcript_txt)
    user_friendly_result = clean_llama_output(raw_result) 
    return user_friendly_result

# Gradio interface:

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn=Summarize_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Meeting Summary App",
                     description="Upload the meeting's audio file")

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
