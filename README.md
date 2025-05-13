# 🎬 Netflix Film Öneri Sistemi

Bu proje, Netflix veri seti kullanılarak (kaggle) geliştirilmiş içerik tabanlı bir film öneri sistemidir. Kullanıcıdan alınan film adına göre, yönetmen, oyuncular, açıklama ve tür gibi metinsel özelliklere dayanarak en benzer 10 filmi önerir.

## 🚀 Canlı Uygulama

Uygulamayı canlı olarak görmek için:  
👉 [https://netflixrecommendation-femeogktn4xzapfshhcxnd.streamlit.app](https://netflixrecommendation-femeogktn4xzapfshhcxnd.streamlit.app)

content_based.py python dosyası ise kullanıcıdan input alarak öneri yapıyor o şekilde de kullanabilirsiniz ama dosya yolunun doğru olduğundan emin olun!

## 📌 Kullanılan Özellikler

- **TF-IDF Vektörleştirme** ile içerik temsili
- **Cosine Similarity** ile benzerlik ölçümü
- **Streamlit** ile etkileşimli web arayüzü
- `app.py`: Streamlit arayüzü
- `content_based.py`: Komut satırından çalışan versiyon

## 📂 Dosya Yapısı

NetflixRecommendation/
├── app.py # Web uygulaması
├── content_based.py # Terminal üzerinden çalışan sürüm
├── netflix_titles.csv # Kullanılan veri seti
└── requirements.txt # Gerekli kütüphaneler

## 🛠️ Kullanılan Teknolojiler

- Python
- pandas, scikit-learn, numpy
- Streamlit
- Git & GitHub

## 👤 Geliştirici

**Melina129**  
[GitHub Profilim](https://github.com/Melina129)

---

Bu proje kişisel öğrenme amaçlı geliştirilmiştir. İçerik önerisi mantığını anlamak ve yayına almak isteyenler için ideal örnektir.
