import warnings
import os
from audio_input import record_audio
from transcriber import transcribe_audio
from contextualizer import extract_key_phrases
from summary import generate_notes

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    audio_file = "input.wav"
    record_audio(filename=audio_file)
    transcription = transcribe_audio(filename=audio_file)
    key_phrases = extract_key_phrases(transcription)
    notes = generate_notes(transcription, key_phrases)

    print("\n--- Final Output ---")
    print("Transcription:\n", transcription)
    print("Key Topics:\n", key_phrases)
    print("Generated Notes:\n", notes)

if __name__ == "__main__":
    main()