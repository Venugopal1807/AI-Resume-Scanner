import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from typing import Dict, List, Set
import string

class NLPAnalyzer:
    def __init__(self):
        # Download all required NLTK data with error handling
        required_packages = [
            'punkt', 
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words',
            'stopwords'
        ]

        for package in required_packages:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                raise Exception(f"Failed to download NLTK package {package}: {str(e)}")

        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            raise Exception(f"Failed to initialize stop words: {str(e)}")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            entities = ne_chunk(tagged)

            extracted = {
                'PERSON': set(),
                'ORGANIZATION': set(),
                'GPE': set()  # Geographical entities
            }

            for chunk in entities:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join(c[0] for c in chunk.leaves())
                    if chunk.label() in extracted:
                        extracted[chunk.label()].add(entity_text)

            return {k: list(v) for k, v in extracted.items()}
        except Exception as e:
            print(f"Warning: Entity extraction failed: {str(e)}")
            return {'PERSON': [], 'ORGANIZATION': [], 'GPE': []}

    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        try:
            tokens = word_tokenize(text.lower())
            # Remove punctuation and stop words
            tokens = [token for token in tokens 
                     if token not in string.punctuation 
                     and token not in self.stop_words
                     and len(token) > 2]
            return list(set(tokens))
        except Exception as e:
            print(f"Warning: Keyword extraction failed: {str(e)}")
            return []

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        try:
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation
            text = ''.join([char for char in text if char not in string.punctuation])
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        except Exception as e:
            print(f"Warning: Text preprocessing failed: {str(e)}")
            return text