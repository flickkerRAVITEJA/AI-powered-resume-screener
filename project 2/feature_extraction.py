import pandas as pd

# Load the datasets
resumes = pd.read_csv('resumes.csv')
jobs = pd.read_csv('job_descriptions.csv')

# Display the first few rows
print("Resumes:")
print(resumes.head())

print("\nJob Descriptions:")
print(jobs.head())
