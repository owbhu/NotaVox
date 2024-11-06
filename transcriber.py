import whisper

def transcribe_audio(filename="input.wav"):
    model = whisper.load_model("base")
    print("Transcribing audio...")
    result = model.transcribe(filename)
    
    transcription = result['text']
    return transcription
