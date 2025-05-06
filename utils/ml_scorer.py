import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import re

class MLScorer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Consider both unigrams and bigrams
            max_features=5000
        )
        
    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from text using regex patterns."""
        # Common programming languages and technologies
        tech_patterns = r'\b(python|java|javascript|c\+\+|ruby|php|sql|html|css|react|angular|vue|node\.js|docker|kubernetes|aws|azure|git|machine learning|deep learning|ai|nlp)\b'
        skills = re.findall(tech_patterns, text.lower())
        return list(set(skills))
    
    def extract_education(self, text: str) -> Dict[str, float]:
        """Extract education level and assign scores."""
        education_scores = {
            'phd': 1.0,
            'master': 0.8,
            'bachelor': 0.6,
            'associate': 0.4
        }
        
        found_scores = []
        text_lower = text.lower()
        
        for level, score in education_scores.items():
            if level in text_lower or f"{level}'s" in text_lower:
                found_scores.append(score)
                
        return max(found_scores) if found_scores else 0.3
    
    def extract_experience(self, text: str) -> float:
        """Extract years of experience."""
        # Pattern to match X years of experience
        pattern = r'(\d+)[\+]?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)'
        matches = re.findall(pattern, text.lower())
        
        if matches:
            years = max([int(y) for y in matches])
            # Normalize experience score (cap at 15 years)
            return min(years / 15.0, 1.0)
        return 0.0
    
    def calculate_advanced_scores(self, job_desc: str, resume: str) -> Dict[str, float]:
        """Calculate detailed scores using multiple criteria."""
        # Content similarity using TF-IDF
        documents = [job_desc, resume]
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        content_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        
        # Skills matching
        job_skills = set(self.extract_skills(job_desc))
        resume_skills = set(self.extract_skills(resume))
        skills_score = len(job_skills.intersection(resume_skills)) / len(job_skills) if job_skills else 0.0
        
        # Education score
        education_score = self.extract_education(resume)
        
        # Experience score
        experience_score = self.extract_experience(resume)
        
        # Calculate weighted scores
        weights = {
            'content_similarity': 0.3,
            'skills_match': 0.3,
            'education': 0.2,
            'experience': 0.2
        }
        
        scores = {
            'content_similarity': round(content_score * 100, 2),
            'skills_match': round(skills_score * 100, 2),
            'education_level': round(education_score * 100, 2),
            'experience_level': round(experience_score * 100, 2),
            'overall_score': round(
                (content_score * weights['content_similarity'] +
                 skills_score * weights['skills_match'] +
                 education_score * weights['education'] +
                 experience_score * weights['experience']) * 100,
                2
            )
        }
        
        # Add matched skills details
        scores['matched_skills'] = list(job_skills.intersection(resume_skills))
        scores['missing_skills'] = list(job_skills - resume_skills)
        
        return scores