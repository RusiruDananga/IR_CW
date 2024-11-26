import streamlit as st
import json
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import urllib.parse

# Set page configuration
st.set_page_config(page_title="PDF Search Engine", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .result-container {
        background-color: #f9f9f9;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .result-title {
        font-size: 18px;
        font-weight: bold;
        color: #2c3e50;
    }
    .result-snippet {
        font-size: 14px;
        color: #7f8c8d;
    }
    .result-score {
        font-size: 12px;
        color: #16a085;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load preprocessed data
def load_preprocessed_data(input_file):
    with open(input_file, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

# Preprocess user query
def preprocess_query(query):
    query = query.lower()  # Lowercasing
    query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
    query = re.sub(r'\d+', '', query)  # Remove numbers
    return query

# Load preprocessed documents
input_file = "preprocessed_data.json"
documents = load_preprocessed_data(input_file)

# Base directory
base_directory = r"D:\NIBM\Masters\Information Retreival\CW\comscds232p007\test"

# Extract document names and content
document_names = [doc["document_name"] for doc in documents]
document_paths = [
    "file:///" + urllib.parse.quote(
        os.path.normpath(os.path.join(base_directory, doc["file_path"])).replace("\\", "/")
    )
    for doc in documents
]
document_contents = [doc["content"] for doc in documents]

# Create TF-IDF vectorizer and fit to document content
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(document_contents)

# Streamlit app
# Sidebar
st.sidebar.title("About")
st.sidebar.markdown(
    """
    This PDF Search Engine allows you to search preprocessed documents using natural language queries. 
    Enter a query in the input box to find relevant documents.
    """
)
st.sidebar.markdown("### Tips:")
st.sidebar.markdown("- Use specific keywords for better results.")
st.sidebar.markdown("- Results are ranked by relevance.")

# Main title
st.title("📄 PDF Search Engine")
st.markdown("Search through preprocessed PDF documents by entering a query below.")

# User query input
user_query = st.text_input("Enter your search query:", placeholder="Type your query here...")

if user_query:
    # Preprocess the query
    processed_query = preprocess_query(user_query)

    # Transform the query into TF-IDF vector
    query_vector = vectorizer.transform([processed_query])

    # Compute cosine similarity between the query and documents
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Rank documents by similarity scores
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_documents = [
        {
            "name": document_names[i],
            "path": document_paths[i],
            "content_snippet": " ".join(document_contents[i].split()[:50]),  # First 50 words
            "score": similarity_scores[i],
        }
        for i in ranked_indices
        if similarity_scores[i] > 0
    ]

    if ranked_documents:
        st.markdown("### 🔍 Search Results:")
        for doc in ranked_documents:
            st.markdown(
                f"""
                <div class="result-container">
                    <p class="result-title"><a href="{doc['path']}" target="_blank">{doc['name']}</a></p>
                    <p class="result-snippet">Snippet: {doc['content_snippet']}...</p>
                    <p class="result-score">Relevance Score: {doc['score']:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown("### No relevant documents found. 😔")
else:
    st.markdown("Enter a query above to search.")