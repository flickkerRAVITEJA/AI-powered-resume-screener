from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load datasets
resumes = pd.read_csv('resumes.csv')
jobs = pd.read_csv('job_descriptions.csv')

# Combine resume fields into a single string for each person
resumes['Combined'] = resumes['Skills'] + ' ' + resumes['Experience'] + ' ' + resumes['Education']
jobs['Combined'] = jobs['Skills Required'] + ' ' + jobs['Education Required']

# Vectorize the text data
vectorizer = TfidfVectorizer()
resume_vectors = vectorizer.fit_transform(resumes['Combined'])
job_vectors = vectorizer.transform(jobs['Combined'])

# Calculate cosine similarity
scores = cosine_similarity(resume_vectors, job_vectors)

# Print matching scores
print("Matching Scores (Resumes vs Jobs):")
print(scores)
