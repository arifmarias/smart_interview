import speech_recognition as sr
import streamlit as st
import numpy as np
import noisereduce as nr
import soundfile as sf
import whisper
import os

# Create a recognizer instance
r = sr.Recognizer()

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
                 
                           
        except:
            st.write("Speech Recognition could not understand audio")
def main():
    st.title("Real-time Speech-to-Text")
    st.button("Start Listening", on_click=listen_and_transcribe)
    

if __name__ == "__main__":
    main()
    if os.path.exists("/Users/m1/Documents/DS Project/smart_interview/temp.wav"):
        os.remove("/Users/m1/Documents/DS Project/smart_interview/temp.wav")
    else:
        print("The file does not exist")
