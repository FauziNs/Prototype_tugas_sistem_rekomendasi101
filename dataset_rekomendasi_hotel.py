# -*- coding: utf-8 -*-
"""dataset_rekomendasi_hotel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GtOV8Tmqns9HCfJh231quUrt5_caEIfA

1. Persiapan Lingkungan
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import subprocess
subprocess.check_call(['pip', 'install', 'streamlit'])

"""2. Preprocessing Data"""

def load_and_preprocess_data(file_path):
    # Load dataset
    hotels_df = pd.read_csv('processed_hotel_data.csv')

    # Menghapus baris dengan nilai penting yang hilang
    hotels_df = hotels_df.dropna(subset=['name', 'city', 'hotelFeatures'])

    # Membuat kolom combined_features
    hotels_df['combined_features'] = hotels_df['name'] + ' ' + hotels_df['city'] + ' ' + \
                                     hotels_df['region'] + ' ' + hotels_df['hotelFeatures']

    # Preprocessing teks
    hotels_df['processed_features'] = hotels_df['combined_features'].apply(preprocess_text)

    return hotels_df

def preprocess_text(text):
    if isinstance(text, str):
        # Konversi ke huruf kecil
        text = text.lower()
        # Hapus karakter khusus dan tanda baca
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenisasi
        words = text.split()
        # Hapus stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        # Stemming
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]
        # Gabungkan kembali
        return ' '.join(words)
    return ''

"""3. Implementasi Content-Based Filtering"""

def create_tfidf_matrix(hotels_df):
    # Inisialisasi TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000)

    # Membuat matriks TF-IDF
    tfidf_matrix = tfidf.fit_transform(hotels_df['processed_features'])

    # Menghitung cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

def get_basic_recommendations(hotel_index, cosine_sim, hotels_df, num_recs=10):
    # Mendapatkan skor kesamaan untuk semua hotel dengan hotel referensi
    sim_scores = list(enumerate(cosine_sim[hotel_index]))

    # Mengurutkan hotel berdasarkan skor kesamaan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ambil top N hotel (kecuali hotel yang sama dengan referensi)
    sim_scores = sim_scores[1:num_recs+1]

    # Ambil indeks hotel
    hotel_indices = [i[0] for i in sim_scores]

    # Kembalikan dataframe dengan rekomendasi
    return hotels_df.iloc[hotel_indices].reset_index(drop=True)

"""4. Implementasi Rekomendasi Berbasis Kelompok"""

def get_group_recommendations(hotel_index, cosine_sim, hotels_df, group_size, num_recs=10):
    # Mendapatkan rekomendasi dasar (lebih banyak untuk pemfilteran)
    basic_recs = get_basic_recommendations(hotel_index, cosine_sim, hotels_df, num_recs * 2)

    # Menentukan faktor harga berdasarkan ukuran kelompok
    if group_size <= 2:
        price_factor = 1.0
    elif group_size <= 4:
        price_factor = 1.5
    elif group_size <= 6:
        price_factor = 2.0
    else:
        price_factor = 2.5

    # Mendapatkan hotel referensi
    ref_hotel = hotels_df.iloc[hotel_index]

    # Menghitung kisaran harga yang sesuai
    if pd.notna(ref_hotel['cheapestRate_perNight_totalFare']):
        target_price = ref_hotel['cheapestRate_perNight_totalFare'] * price_factor
        min_price = target_price * 0.7
        max_price = target_price * 1.3

        # Filter berdasarkan kisaran harga
        filtered_recs = basic_recs[
            (basic_recs['cheapestRate_perNight_totalFare'] >= min_price) &
            (basic_recs['cheapestRate_perNight_totalFare'] <= max_price)
        ]
    else:
        # Jika tidak ada informasi harga, gunakan rekomendasi dasar
        filtered_recs = basic_recs

    # Jika tidak cukup rekomendasi setelah pemfilteran, ambil dari rekomendasi dasar
    if len(filtered_recs) < num_recs:
        filtered_recs = basic_recs

    # Urutkan berdasarkan peringkat pengguna dan bintang
    filtered_recs = filtered_recs.sort_values(by=['userRating', 'starRating'], ascending=False)

    # Ambil N teratas
    return filtered_recs.head(num_recs).reset_index(drop=True)

"""5. Implementasi Rekomendasi Lanjutan dengan Preferensi Kelompok"""

def get_advanced_group_recommendations(hotel_index, cosine_sim, hotels_df, group_composition, preferences, num_recs=10):
    # group_composition = {'adults': 2, 'children': 2, 'seniors': 1}
    # preferences = {'pool': 0.8, 'restaurant': 0.6, 'wifi': 0.9}

    # Menghitung ukuran kelompok
    group_size = sum(group_composition.values())

    # Mendapatkan rekomendasi dasar berbasis kelompok
    basic_group_recs = get_group_recommendations(hotel_index, cosine_sim, hotels_df, group_size, num_recs * 2)

    # Fungsi untuk menghitung skor preferensi
    def calculate_preference_score(hotel_features):
        score = 0
        for pref, weight in preferences.items():
            if pref.lower() in str(hotel_features).lower():
                score += weight
        return score / len(preferences) if preferences else 0

    # Fungsi untuk menghitung skor demografis
    def calculate_demographic_score(hotel_features):
        score = 0
        if group_composition.get('children', 0) > 0:
            child_keywords = ['kids', 'family', 'playground', 'children']
            for keyword in child_keywords:
                if keyword in str(hotel_features).lower():
                    score += 1
            score = score / len(child_keywords) * group_composition['children'] / group_size

        if group_composition.get('seniors', 0) > 0:
            senior_keywords = ['accessible', 'elevator', 'quiet', 'assistance']
            for keyword in senior_keywords:
                if keyword in str(hotel_features).lower():
                    score += 1
            score += score / len(senior_keywords) * group_composition['seniors'] / group_size

        return score

    # Menghitung skor komposit
    basic_group_recs['preference_score'] = basic_group_recs['hotelFeatures'].apply(calculate_preference_score)
    basic_group_recs['demographic_score'] = basic_group_recs['hotelFeatures'].apply(calculate_demographic_score)
    basic_group_recs['composite_score'] = basic_group_recs['preference_score'] * 0.6 + basic_group_recs['demographic_score'] * 0.4

    # Urutkan berdasarkan skor komposit
    advanced_recs = basic_group_recs.sort_values(by='composite_score', ascending=False)

    # Ambil N teratas
    return advanced_recs.head(num_recs).reset_index(drop=True)