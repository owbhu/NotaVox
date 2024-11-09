import os
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def generate_notes(text, key_phrases, max_length=500):
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If an OpenAI API key is available, use GPT-3.5 for note generation
    if api_key:
        openai.api_key = api_key
        prompt = (
            f"Generate comprehensive notes based on the following text and topics.\n\n"
            f"Text: {text}\n"
            f"Key Topics: {', '.join(key_phrases)}\n\n"
            "Organize the notes by main points and include any necessary details."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_length,
            temperature=0.5,
        )
        notes = response['choices'][0]['message']['content']
    else:
        # Fallback to FLAN-T5 if no API key is found
        print("OpenAI API key not found. Using FLAN-T5 as a fallback.")
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        generator = pipeline("text2text-generation", model="google/flan-t5-large", device=device)

        # Construct the prompt for FLAN-T5
        prompt = (
            f"Summarize the main points based on these topics. Provide organized and detailed notes.\n\n"
            f"Text: {text}\n"
            f"Key Topics: {', '.join(key_phrases)}\n\n"
            "Notes:"
        )
        notes = generator(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]

    return notes