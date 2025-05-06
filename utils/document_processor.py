import PyPDF2
import docx
import io
from typing import List, Dict, Union
from pathlib import Path

class DocumentProcessor:
    @staticmethod
    def extract_text(file) -> str:
        """Extract text from uploaded document file."""
        try:
            # Get file extension
            filename = file.name.lower()

            # Reset file pointer to beginning
            file.seek(0)

            if filename.endswith('.pdf'):
                return DocumentProcessor._extract_from_pdf(file)
            elif filename.endswith('.docx'):
                return DocumentProcessor._extract_from_docx(file)
            else:
                raise ValueError(f"Unsupported file format: {Path(filename).suffix}")

        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    @staticmethod
    def _extract_from_pdf(file) -> str:
        """Extract text from PDF file."""
        try:
            # Create BytesIO object from file content
            file_bytes = io.BytesIO(file.read())

            pdf_reader = PyPDF2.PdfReader(file_bytes)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
        finally:
            file_bytes.close()

    @staticmethod
    def _extract_from_docx(file) -> str:
        """Extract text from DOCX file."""
        try:
            # Create BytesIO object from file content
            file_bytes = io.BytesIO(file.read())

            doc = docx.Document(file_bytes)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text).strip()
        except Exception as e:
            raise Exception(f"Error processing DOCX: {str(e)}")
        finally:
            file_bytes.close()

    @staticmethod
    def get_document_stats(text: str) -> Dict:
        """Calculate basic document statistics."""
        try:
            words = text.split()
            sentences = text.split('.')
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
                'char_count': len(text)
            }
        except Exception as e:
            print(f"Warning: Failed to calculate document stats: {str(e)}")
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'char_count': 0
            }