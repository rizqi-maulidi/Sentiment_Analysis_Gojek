# Dokumentasi Analisis Sentimen Ulasan Gojek

## Deskripsi Proyek

Proyek ini bertujuan untuk melakukan analisis sentimen terhadap ulasan aplikasi Gojek menggunakan berbagai teknik machine learning dan deep learning. Sistem ini dapat mengklasifikasikan ulasan menjadi tiga kategori sentimen: positif, negatif, dan netral.

## Fitur Utama

- **Web Scraping**: Mengumpulkan data ulasan dari Google Play Store
- **Text Preprocessing**: Pembersihan dan normalisasi teks berbahasa Indonesia
- **Sentiment Analysis**: Klasifikasi sentimen menggunakan lexicon-based approach
- **Deep Learning Models**: Implementasi LSTM dan Transformer untuk prediksi sentimen
- **Visualisasi Data**: Word cloud dan distribusi sentimen

## Arsitektur Sistem

### 1. Data Collection (Web Scraping)

```python
import pandas as pd

# Konversi hasil scraping ke DataFrame
df = pd.DataFrame(scrapreview)

# Pilih kolom yang relevan
df = df[['content', 'score', 'at', 'thumbsUpCount']]  
df.rename(columns={'content': 'Review', 'score': 'Rating', 'at': 'Tanggal', 'thumbsUpCount': 'Likes'}, inplace=True)

# Simpan ke file CSV
df.to_csv('ulasan_gojek.csv', index=False, encoding='utf-8')
```

**Output**: File CSV berisi ulasan Gojek dengan kolom Review, Rating, Tanggal, dan Likes.

### 2. Data Preprocessing

#### 2.1 Text Cleaning Pipeline

Proses pembersihan teks meliputi:

1. **Cleaning**: Menghapus mention, hashtag, URL, angka, dan karakter khusus
2. **Case Folding**: Mengubah teks menjadi huruf kecil
3. **Slang Word Normalization**: Mengganti kata-kata slang dengan kata standar
4. **Tokenization**: Memecah teks menjadi token
5. **Stopword Removal**: Menghapus kata-kata yang tidak bermakna
6. **Stemming**: Mengubah kata ke bentuk dasar (menggunakan Sastrawi)

#### 2.2 Komponen Preprocessing

- **Custom Stopwords**: Menggunakan stopwords dari NLTK (Indonesia & Inggris) + kamus kustom
- **Slang Dictionary**: Kamus kata-kata tidak baku ke bentuk baku
- **Sastrawi Stemmer**: Library stemming untuk bahasa Indonesia

### 3. Sentiment Labeling

#### 3.1 Lexicon-Based Approach

Menggunakan kamus sentimen Indonesia untuk memberikan label otomatis:

- **Positive Lexicon**: Kata-kata dengan nilai positif
- **Negative Lexicon**: Kata-kata dengan nilai negatif
- **Scoring System**: 
  - Score > 0: Positif
  - Score < 0: Negatif  
  - Score = 0: Netral

#### 3.2 Distribusi Sentimen

Hasil pelabelan menunjukkan distribusi sentimen dalam dataset yang dapat divisualisasikan menggunakan pie chart dan bar chart.

### 4. Machine Learning Models

Proyek ini mengimplementasikan tiga model berbeda:

#### Model 1: LSTM + TF-IDF
- **Arsitektur**: LSTM dengan 2 layer (128 dan 64 unit)
- **Feature Extraction**: TF-IDF Vectorizer (max 5000 features)
- **Data Split**: 70/30
- **Dropout**: 0.5 untuk mencegah overfitting
- **Akurasi**: Training 96.66%, Testing 91.62%

#### Model 2: Bidirectional LSTM + Word2Vec
- **Arsitektur**: Bidirectional LSTM dengan embedding layer
- **Feature Extraction**: Word2Vec (100 dimensi)
- **Data Split**: 80/20
- **Early Stopping**: Monitor validation loss dengan patience=3
- **Akurasi**: Training 85.86%, Testing 83.56%

#### Model 3: Transformer
- **Arsitektur**: Custom Transformer dengan Multi-Head Attention
- **Components**: 
  - Embedding Layer
  - Multi-Head Attention (2 heads)
  - Feed Forward Network
  - Layer Normalization
  - Global Average Pooling
- **Data Split**: 80/20
- **Akurasi**: Training 97.63%, Testing 92.41%

## Instalasi dan Dependencies

### Requirements

```python
pandas
numpy
matplotlib
seaborn
nltk
tensorflow
scikit-learn
wordcloud
gensim
sastrawi
requests
```

### Setup Environment

```bash
pip install pandas numpy matplotlib seaborn nltk tensorflow scikit-learn wordcloud gensim sastrawi requests
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Penggunaan

### 1. Data Preparation

```python
# Load dataset
app_review = pd.read_csv('ulasan_gojek.csv')
app_reviews_df = pd.DataFrame(app_review)

# Clean data
clean_df = app_reviews_df.dropna()
clean_df = clean_df.drop_duplicates()
```

### 2. Text Preprocessing

```python
# Apply preprocessing pipeline
clean_df['text_clean'] = clean_df['Review'].apply(cleaningText)
clean_df['text_casefoldingText'] = clean_df['text_clean'].apply(casefoldingText)
clean_df['text_slangwords'] = clean_df['text_casefoldingText'].apply(fix_slangwords)
clean_df['text_tokenizingText'] = clean_df['text_slangwords'].apply(tokenizingText)
clean_df['text_stopword'] = clean_df['text_tokenizingText'].apply(filteringText)
clean_df['text_akhir'] = clean_df['text_stopword'].apply(toSentence)
```

### 3. Sentiment Analysis

```python
# Apply lexicon-based sentiment analysis
results = clean_df['text_stopword'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
clean_df['polarity_score'] = results[0]
clean_df['polarity'] = results[1]
```

### 4. Model Training

```python
# Example for Transformer model
model3 = build_transformer_model(max_len, vocab_size, num_classes=num_classes)
model3.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)
```

### 5. Inference

```python
# Prediksi sentimen untuk teks baru
new_texts = [
    "Saya sangat puas dengan produk ini",
    "Gojek biasa aja", 
    "Pengalaman dengan driver sangat buruk"
]

new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=max_len, padding='post')
pred_probs = model3.predict(new_padded)
pred_classes = np.argmax(pred_probs, axis=1)
pred_labels = label_encoder.inverse_transform(pred_classes)
```

## Visualisasi

### 1. Word Cloud
- Word cloud per kategori sentimen (positif, negatif, netral)  
![image](https://github.com/user-attachments/assets/4dbe158e-3977-4a65-a315-d0101b36c4c1)


### 2. Data Analysis
- Distribusi kelas sentimen  
![image](https://github.com/user-attachments/assets/ba0fb5e9-85ff-42ba-9ceb-274bca9ed65b)

- Distribusi panjang teks  
![image](https://github.com/user-attachments/assets/a88574fc-a877-4b50-9597-a3d7e789ae03)

- Kata-kata paling sering muncul (berdasarkan TF-IDF)  
![image](https://github.com/user-attachments/assets/cabe71e9-bf11-4ff5-948c-65bb88b27f2d)


## Hasil Evaluasi

| Model   | Arsitektur      | Ekstraksi Fitur  | Pembagian Data | Akurasi Training | Akurasi Testing |
|---------|------------------|------------------|----------------|------------------|-----------------|
| Model 1 | LSTM             | TF-IDF           | 70/30          | 96.66%           | 91.62%          |
| Model 2 | LSTM | Word2Vec       | 80/20          | 85.86%           | 83.56%          |
| Model 3 | Transformer      | Embedding        | 80/20          | 97.63%           | 92.41%          |

## Kesimpulan

Model Transformer (Model 3) menunjukkan performa terbaik dengan akurasi testing 92.41%, diikuti oleh LSTM dengan TF-IDF (Model 1) dengan akurasi 91.62%. Model ini efektif untuk mengklasifikasikan sentimen ulasan aplikasi Gojek dalam bahasa Indonesia.

## Kontribusi

Proyek ini dapat dikembangkan lebih lanjut dengan:
- Menambah variasi dataset
- Implementasi model pre-trained (BERT Indonesia)
- Optimasi hyperparameter
- Deployment ke production environment

## Lisensi

Proyek ini dibuat untuk tujuan edukasi dan penelitian.

---

*Dokumentasi ini menjelaskan implementasi lengkap sistem analisis sentimen untuk ulasan aplikasi Gojek menggunakan berbagai teknik machine learning dan deep learning.*
