## This is a dev and fun project for Smart Interview
Using openai whisper speech2text conversion
### Setup
#### Setup your Audio
1. Use VB-Cable
2. Download from https://vb-audio.com/Cable/
3. In Mac Spotlight to "Audio MIDI Setup"
4. Create "Aggregate Device" and select "Mac Microphone" and "VB-Cable"
5. Multi-Output Device select "Mac Speaker" and "VB-Cable"

#### Setup your Environment 
1. pip install -U openai-whisper
on MacOS using Homebrew (https://brew.sh/)
2. brew install ffmpeg
3. whisper model use "small" (fairly good transcriber)
4. small piece of code to use the whisper
```
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```