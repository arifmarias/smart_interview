import speech_recognition as sr
import streamlit as st
import numpy as np
import noisereduce as nr
import soundfile as sf
import whisper
import os
import google.generativeai as genai
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession


# Create a recognizer instance
r = sr.Recognizer()
API_KEY = ''
genai.configure(api_key = API_KEY)
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat()
def listen_and_transcribe():
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        st.write("Listening...")
        # Read the audio data from the default microphone
        audio_data = r.listen(source)
        np_audio_data = np.frombuffer(audio_data.frame_data, np.int16)
        reduced_noise = nr.reduce_noise(y=np_audio_data, sr=audio_data.sample_rate, y_noise=np_audio_data)
        
        sf.write('temp.wav', reduced_noise, audio_data.sample_rate)
        # Load Whisper Model
        model = whisper.load_model("small")
        st.write("Recognizing...")
        # Convert speech to text
        try:
            with sr.AudioFile('temp.wav') as source:
                audio_data = "/Users/m1/Documents/DS Project/smart_interview/temp.wav"
                result = model.transcribe(audio_data, fp16=False)
                st.write(result["text"])
                st.write("Get Response from GenAI")
                st.write(get_chat_response(chat,result["text"]))                  
        except:
            st.write("Speech Recognition could not understand audio")


def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)


def main():
    st.title("Real-time Speech-to-Text")
    st.button("Start Listening", on_click=listen_and_transcribe)
    

if __name__ == "__main__":
    main()
    if os.path.exists("/Users/m1/Documents/DS Project/smart_interview/temp.wav"):
        os.remove("/Users/m1/Documents/DS Project/smart_interview/temp.wav")
    else:
        print("The file does not exist")