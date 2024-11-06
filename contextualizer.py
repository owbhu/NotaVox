from transformers import pipeline
from collections import Counter
import re

def extract_key_phrases(text, top_n=5):
    # Using transformers' NLP pipeline for keyphrase extraction
    keyphrase_extractor = pipeline("feature-extraction")
    words = re.findall(r'\b\w+\b', text.lower())
    common_words = Counter(words).most_common(top_n)
    key_phrases = [word for word, _ in common_words]
    print("Key Topics:", key_phrases)
    return key_phrases
