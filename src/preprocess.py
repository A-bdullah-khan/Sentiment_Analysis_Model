import string

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Necessary NLTK datasets download
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class TextPreprocessor:
    """
    Cleans raw text input: strips HTML, punctuation, converts to lowercase,
    filters stopwords (preserving negations), and applies WordNet Lemmatization.
    """

    def __init__(self):
        self.punctuation = set(string.punctuation)
        # NLTK stopwords set, keeping negations that flip sentiment ("not good")
        self.negations = {"not", "no", "nor", "never", "neither", "cannot"}
        self.stop_words = set(stopwords.words("english")) - self.negations
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove punctuation & convert to lowercase
        text = "".join([char.lower() for char in text if char not in self.punctuation])

        # Tokenize and lemmatize non-stopwords
        tokens = word_tokenize(text)
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        return " ".join(cleaned_tokens)
