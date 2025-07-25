# Install spaCy and download model
!pip install -q spacy
!python -m spacy download en_core_web_sm
# Now import libraries
import spacy
from nltk.stem import PorterStemmer
# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
# Sample text
text = "Text mining is the process of deriving meaningful information from natural language
text."
# Process the text
doc = nlp(text)
# Tokenize and remove stopwords using spaCy
tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
# Apply stemming using NLTK's PorterStemmer
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]
# Output results
print("Original Text:\n", text)
print("\nTokens after Stop Word Removal:\n", tokens)
print("\nStemmed Tokens:\n", stemmed_tokens)
