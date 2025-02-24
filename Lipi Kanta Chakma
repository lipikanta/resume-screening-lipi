import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pytesseract
from PIL import Image
import io
import re

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Clean text by removing spaces, numbers, and stopwords."""
    text = re.sub(r'\d+', '', text)  # Remove numbers
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens) if tokens else "No valid text"

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return preprocess_text(text)

def extract_text_from_image(file):
    """Extract text from an image file using OCR."""
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return preprocess_text(text)

def extract_text_from_csv(file):
    """Extract text from a CSV file."""
    df = pd.read_csv(file)
    text = " ".join(df.astype(str).values.flatten())
    return preprocess_text(text)

def extract_text_from_txt(file):
    """Extract text from a TXT file."""
    text = file.read().decode("utf-8")
    return preprocess_text(text)

def rank_resumes(resumes, job_description):
    """Rank resumes based on similarity to job description using TF-IDF and Cosine Similarity."""
    processed_resumes = [preprocess_text(resume) for resume in resumes]
    processed_resumes = [res for res in processed_resumes if res != "No valid text"]
    
    if not processed_resumes:
        return []
    
    all_texts = [job_description] + processed_resumes
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(vectors[0], vectors[1:]).flatten()
    ranked_resumes = sorted(zip(resumes, similarity_scores), key=lambda x: x[1], reverse=True)
    return ranked_resumes[:10]  # Return top 10 resumes

def main():
    st.title("AI-Powered Resume Screening & Ranking System")
    st.write("Upload resumes in PDF, PNG, TEXT, JPG, JPEG, CSV formats and provide a job description.")
    
    job_description = st.text_area("Enter Job Description")
    uploaded_files = st.file_uploader("Upload Resume Files", accept_multiple_files=True, type=["pdf", "png", "txt", "jpg", "jpeg", "csv"])
    
    if st.button("Rank Resumes") and uploaded_files and job_description.strip():
        resumes = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                resumes.append(extract_text_from_pdf(file))
            elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
                resumes.append(extract_text_from_image(file))
            elif file.type == "text/csv":
                resumes.append(extract_text_from_csv(file))
            elif file.type == "text/plain":
                resumes.append(extract_text_from_txt(file))
        
        ranked_resumes = rank_resumes(resumes, job_description)
        
        if not ranked_resumes:
            st.warning("No valid resumes found. Ensure files contain readable text.")
        else:
            st.subheader("Top 10 Ranked Resumes")
            for idx, (resume, score) in enumerate(ranked_resumes):
                st.write(f"### Rank {idx + 1} - Score: {score:.2f}")
                st.text_area(f"Resume {idx + 1}", resume, height=150)

if __name__ == "__main__":
    main()
