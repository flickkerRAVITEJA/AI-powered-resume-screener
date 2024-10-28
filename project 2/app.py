from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        resume_text = request.form['resume']
        job_text = request.form['job']

        # Vectorize input
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, job_text])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]

        return f"<h1>Match Score: {score:.2f}</h1>"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
