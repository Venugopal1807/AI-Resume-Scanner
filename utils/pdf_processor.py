import PyPDF2
import io
from typing import List, Dict

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    @staticmethod
    def get_document_stats(text: str) -> Dict:
        """Calculate basic document statistics."""
        words = text.split()
        sentences = text.split('.')
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'char_count': len(text)
        }