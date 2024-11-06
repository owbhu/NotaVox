import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="input.wav", sample_rate=16000):
    print("Recording... Press Ctrl+C to stop.")
    try:
        # Record for an extended period, allowing a manual stop
        audio = sd.rec(int(10 * 60 * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording completes or is interrupted
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        write(filename, sample_rate, audio)
        print(f"Audio saved as {filename}")
