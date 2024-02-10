import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os
import hashlib
import openai

client = openai.OpenAI()

# --- File Paths ---
CACHE_DIR = "data/embeddings_cache"
INPUT_DATA_FILE = "sentences-processed.txt"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- Utility Functions ---
def hash_text(text):
    """Returns a SHA256 hash of the given text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_cached_embedding_path(sentence):
    """Generates a file path for a cached embedding based on the sentence hash."""
    hash_id = hash_text(sentence)
    return os.path.join(CACHE_DIR, f"{hash_id}.npy")

def generate_or_load_embedding(sentence):
    """Generates an embedding for a sentence, using cache if available."""
    cache_path = get_cached_embedding_path(sentence)
    if os.path.exists(cache_path):
        embedding = np.load(cache_path)
    else:
        response = client.embeddings.create(
            input=sentence.strip(),  # Ensure lines are stripped of whitespace
            model="text-embedding-3-small"
        )
        embedding = np.array(response.data[0].embedding)
        np.save(cache_path, embedding)
    return embedding

# --- Data Preparation ---
def load_data(input_file, max_lines=5):
    """Loads text lines and generates embeddings. Checks for cached embeddings first, considering max_lines."""
    embeddings = []
    sentences = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break  # Limit to max_lines
            sentence = line.strip()
            sentences.append(sentence)
            embeddings.append(generate_or_load_embedding(sentence))
    return np.array(embeddings), sentences

# --- Training the PCA Model ---
def train_pca(data, n_components=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)
    return pca

# --- Projecting Data and Visualization ---
def project_and_visualize(pca, data_from_file, embeddings_from_input, common_corpus_sentences, user_sentences):
    # Transform both sets of data
    data_from_file_transformed = pca.transform(data_from_file)
    embeddings_from_input_transformed = pca.transform(embeddings_from_input)
    
    # Combine transformed data for plotting
    combined_transformed_data = np.vstack((data_from_file_transformed, embeddings_from_input_transformed))
    
    # Create a color array, differentiating between data_from_file (e.g., 'blue') and embeddings_from_input (e.g., 'red')
    colors = ['Common corpus'] * len(data_from_file_transformed) + ['User sentences'] * len(embeddings_from_input_transformed)
    
    # Create hover text for each point, including common corpus sentences and user sentences
    hover_text = [f"Common sentence {i+1}: {sentence}" for i, sentence in enumerate(common_corpus_sentences)] + \
                 [f"User sentence {i+1}: {sentence}" for i, sentence in enumerate(user_sentences)]
    
    fig = px.scatter_3d(
        x=combined_transformed_data[:, 0],
        y=combined_transformed_data[:, 1],
        z=combined_transformed_data[:, 2],
        color=colors,  # Use the color array for coloring points
        labels={'color': 'Source'},
        color_discrete_map={'Common corpus': 'blue', 'User sentences': 'red'},
        hover_name=hover_text  # Use hover text for each point
    )
    return fig

# --- Streamlit App ---
st.title("PCA Dimensionality Reduction of OpenAI embeddings")

# Streamlit slider for loading data from file
number_of_lines = st.number_input("Number of lines to load from common sentence corpus", min_value=1, max_value=1000, value=5)

# Load text data, generate embeddings if needed, and retrieve common corpus sentences
st.subheader("Data Loading")
data_from_file, common_corpus_sentences = load_data(INPUT_DATA_FILE, number_of_lines)

# Text input for sentences
sentences_input = st.text_area("Enter sentences (one per line)", "The quick brown fox jumps over the lazy dog")
user_sentences = sentences_input.split('\n')

# Generate or load embeddings for entered sentences
with st.spinner("Generating embeddings..."):
    embeddings_from_input = np.array([generate_or_load_embedding(sentence) for sentence in user_sentences if sentence.strip()])

# Train PCA model with combined embeddings
st.subheader("PCA Training")
combined_embeddings = np.vstack((data_from_file, embeddings_from_input))
with st.spinner("Fitting PCA..."):
    pca = train_pca(combined_embeddings)

# Visualization
st.subheader("PCA plot n=3 dimensions")
fig = project_and_visualize(pca, data_from_file, embeddings_from_input, common_corpus_sentences, user_sentences)
st.plotly_chart(fig, use_container_width=True)  # Display Plotly chart
