# 📝 Meeting Summary App
This application provides a streamlined interface to automatically transcribe and summarize meeting audio files using IBM Watsonx.ai, Meta's LLaMA 3, and Whisper from OpenAI. Users can upload an audio file and receive a structured, concise summary of its content.

## 🚀 Features
🎙️ Speech-to-Text: Uses OpenAI's Whisper (tiny.en) model to transcribe audio.

🧠 Summarization: Utilizes Meta's LLaMA 3 model hosted on IBM Watsonx.ai to generate detailed summaries from transcripts.

🖥️ Web Interface: Built with Gradio for a simple, intuitive UI—no coding required.

## 🔧 Technologies Used
LangChain for the orchestration of language models.

Transformers (pipeline) for speech recognition.

Gradio for the interactive UI.

IBM Watsonx.ai as the backend for LLM summarization using the LLaMA 3 model.

Regex for output post-processing.

## 🧪 How It Works
Upload your .wav or .mp3 audio file through the Gradio UI.

### The app:

Transcribes the audio using Whisper.

Sends the transcript to the LLaMA 3 model with a structured prompt.

Cleans the model output for readability.

Displays a neatly formatted summary of the key points in the meeting.

## 🛠️ Environment Requirements
This project uses the following Python packages:

torch

transformers

gradio

langchain

ibm-watson-machine-learning

Note: Since this app uses IBM Watsonx.ai services, ensure valid credentials and a project_id are available.

## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more information.

## 📌 Notes
The app currently uses the openai/whisper-tiny.en model for transcription. For better accuracy, consider switching to a larger Whisper variant.

Prompting and summary logic is defined within the script and can be customized for domain-specific use cases.


