#Gerekli Kütüphaneleri İçe Aktar
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Veri dosyasını oku
df = pd.read_csv("netflix_titles.csv")

#Content-Based Öneri İçin Gerekli Sütunları Seç
features = ['director', 'cast', 'listed_in', 'description']

# Eksik alanları boş string yap (çıkarmıyoruz)
for feature in features:
    df[feature] = df[feature].fillna('')

# Gereksiz boşlukları da temizle
for feature in features:
    df[feature] = df[feature].str.strip()
    df[feature] = df[feature].str.replace(r'\s+', ' ', regex=True)

# Özellikleri birleştiren fonksiyon
def combine_features(row):
    return ' '.join([row[feature] for feature in features])

# Her satır için birleştirme işlemi
df['combined_features'] = df.apply(combine_features, axis=1)

# TF-IDF vektörleştirici nesnesi
tfidf = TfidfVectorizer(stop_words='english')

# combined_features sütununu vektörleştir
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Benzerlik matrisini hesapla
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# NumPy'ın gösterim ayarını genişlet
# np.set_printoptions(threshold=np.inf)

def recommend(title, df, cosine_sim):
    # Filmin index'ini bul
    indices = df[df['title'].str.lower() == title.lower()].index

    if len(indices) == 0:
        print("Cannot find the movie. Please check the title.")
        return

    idx = indices[0]

    # Benzerlik skorlarını al
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Skora göre sırala (yüksekten düşüğe)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Kendisi dışındaki ilk 10 benzer filmi al
    sim_scores = sim_scores[1:11]

    # Film adlarını yazdır
    print(f"'{df['title'][idx]}' için öneriler:\n")
    for i, score in sim_scores:
        print(f"{df['title'][i]}  (Benzerlik: {score:.3f})")

# Kullanıcıdan film adı al
user_input = input("Film adı girin: ")

# Öneri yap
recommend(user_input, df, cosine_sim)

