import sounddevice as sd
from scipy.io.wavfile import write
import whisper

def record_audio(duration, filename="input.wav", sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(filename, sample_rate, audio)  # Save as WAV file
    print("Recording complete and saved as", filename)

def transcribe_audio(filename="input.wav"):
    # Load the Whisper model
    model = whisper.load_model("base")
    
    # Transcribe the audio file
    print("Transcribing audio...")
    result = model.transcribe(filename)
    text = result['text']
    print("Transcription:", text)
    return text

# Usage
duration = 10  # Record for 10 seconds
record_audio(duration, filename="input.wav")
transcription = transcribe_audio(filename="input.wav")
