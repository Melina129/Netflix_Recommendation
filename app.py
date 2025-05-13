# Gerekli Kütüphaneleri İçe Aktar
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Custom Netflix tarzı arayüz teması
st.markdown(
    """
    <style>
    body {
        background-color: #141414;
        color: white;
    }
    .stApp {
        background-color: #141414;
    }
    .css-1d391kg, .css-18ni7ap, .css-1v3fvcr, .css-1kyxreq {
        background-color: #141414;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #333333;
        color: white;
        border: 1px solid #e50914;
    }
    .stButton > button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit Başlık
st.title("🎬 Netflix İçerik Tabanlı Film Öneri Sistemi")

# TF-IDF ve Cosine Similarity önbellekli hesaplama fonksiyonu
@st.cache_resource
def load_data_and_similarity():
    df = pd.read_csv("netflix_titles.csv")
    features = ['director', 'cast', 'listed_in', 'description']
    for feature in features:
        df[feature] = df[feature].fillna('').str.strip().str.replace(r'\s+', ' ', regex=True)
    df['combined_features'] = df.apply(lambda row: ' '.join([row[feature] for feature in features]), axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

# Veriyi ve benzerlik matrisini yükle
df, cosine_sim = load_data_and_similarity()

# Öneri fonksiyonu
def recommend(title, df, cosine_sim):
    indices = df[df['title'].str.lower() == title.lower()].index
    if len(indices) == 0:
        return ["Film bulunamadı. Lütfen doğru yazıldığından emin olun."]
    idx = indices[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    return [df['title'][i] for i, score in sim_scores]

# Kullanıcı girişi ve öneri gösterimi
user_input = st.text_input("Bir film adı girin:")

if st.button("Önerileri Göster"):
    results = recommend(user_input, df, cosine_sim)
    st.subheader("Önerilen Filmler:")
    for r in results:
        st.write("• " + r)
