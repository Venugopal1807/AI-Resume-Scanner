from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple

class ResumeRanker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def calculate_similarity(self, job_description: str, resumes: List[str]) -> List[float]:
        """Calculate similarity scores between job description and resumes."""
        # Combine job description and resumes for vectorization
        documents = [job_description] + resumes
        
        # Calculate TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Calculate similarity between job description and each resume
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        return similarities[0]

    def calculate_score_breakdown(self, job_desc: str, resume: str) -> Dict[str, float]:
        """Calculate detailed score breakdown for a resume."""
        # Content similarity using TF-IDF
        content_score = float(self.calculate_similarity(job_desc, [resume])[0])
        
        # Keyword matching score
        job_keywords = set(job_desc.lower().split())
        resume_keywords = set(resume.lower().split())
        keyword_match = len(job_keywords.intersection(resume_keywords)) / len(job_keywords)
        
        # Length score (penalize if too short or too long)
        ideal_length = len(job_desc.split()) * 1.5
        actual_length = len(resume.split())
        length_score = 1 - min(abs(actual_length - ideal_length) / ideal_length, 0.5)
        
        return {
            'content_similarity': round(content_score * 100, 2),
            'keyword_match': round(keyword_match * 100, 2),
            'length_score': round(length_score * 100, 2),
            'overall_score': round((content_score * 0.5 + keyword_match * 0.3 + length_score * 0.2) * 100, 2)
        }