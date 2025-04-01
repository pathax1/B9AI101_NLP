# NLP + Topic Modeling with Streamlit (No BERTopic – PyCharm Version)

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import os
import warnings

from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# --------------------------------
# 🔧 Fix: NLTK Setup for Windows
# --------------------------------
warnings.filterwarnings("ignore")

nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)

nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)
nltk.download("omw-1.4", download_dir=nltk_data_dir)
_ = PunktSentenceTokenizer()

# --------------------------------
# 📘 Helper Functions
# --------------------------------

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def extract_reviews(df):
    return df['review_body'].dropna().tolist()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.lower().split()  # 🚫 DO NOT use word_tokenize
    clean_tokens = [stemmer.stem(w.strip()) for w in tokens if w not in stop_words]
    return clean_tokens


def get_tfidf_matrix(reviews):
    tfidf = TfidfVectorizer(
        max_df=0.99,
        min_df=0.01,
        max_features=1000,
        tokenizer=preprocess_text
    )
    tfidf_matrix = tfidf.fit_transform(reviews)
    return tfidf_matrix, tfidf

def generate_wordcloud(text_data):
    all_text = " ".join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    return wordcloud

def perform_lda(matrix, feature_names, num_topics=5, num_words=10):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(matrix)
    topics = []
    for idx, topic in enumerate(lda.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(f"Topic {idx+1}: " + ", ".join(topic_words))
    return topics

def plot_tsne(matrix, labels=5):
    model = KMeans(n_clusters=labels, random_state=42)
    cluster_labels = model.fit_predict(matrix)
    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    reduced_data = tsne.fit_transform(matrix.toarray())
    return reduced_data, cluster_labels

# --------------------------------
# 🖥️ Streamlit App Interface
# --------------------------------

st.set_page_config(page_title="NLP & Topic Modeling App", layout="wide")
st.title("🧠 NLP & Topic Modeling Dashboard (PyCharm Edition)")

uploaded_file = st.file_uploader("📤 Upload a CSV file with a 'review_body' column", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    # fallback to local file if running offline
    default_path = "Review_data.csv"
    if os.path.exists(default_path):
        st.warning("📎 Using local 'Review_data.csv' since no file was uploaded.")
        df = load_data(default_path)
    else:
        st.error("❌ No file uploaded and no local file found!")
        st.stop()

reviews = extract_reviews(df)

st.subheader("🔍 Sample Reviews")
st.write(reviews[:3])

# WordCloud
st.subheader("☁️ WordCloud")
wc = generate_wordcloud(reviews)
fig, ax = plt.subplots()
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# TF-IDF
tfidf_matrix, tfidf_model = get_tfidf_matrix(reviews)
st.success(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# LDA Topics
st.subheader("📚 Topics from LDA")
feature_names = tfidf_model.get_feature_names_out()
topics = perform_lda(tfidf_matrix, feature_names)
for t in topics:
    st.markdown(f"- {t}")

# t-SNE Visualization
st.subheader("🔢 t-SNE Cluster Visualization")
reduced_data, cluster_labels = plot_tsne(tfidf_matrix)
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis')
st.pyplot(fig2)

st.success("✅ Analysis Complete!")
