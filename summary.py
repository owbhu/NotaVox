from transformers import pipeline
import torch

def generate_notes(text, key_phrases, max_length=500):
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text2text-generation", model="t5-large", device=device)

    # Construct prompt to guide the model towards generating comprehensive notes
    prompt = (
        f"Take the following text and key topics to write detailed and organized notes:\n\n"
        f"Text: {text}\n"
        f"Key Topics: {', '.join(key_phrases)}\n\n"
        f"Organize the notes by main points and add any necessary detail."
    )
    if len(text) > 500:
        text = text[:500] + "..." 

    print("Generating comprehensive notes...")
    notes = generator(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]
    return notes
