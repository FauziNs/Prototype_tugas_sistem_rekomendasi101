import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Sistem Rekomendasi Hotel Berbasis Kelompok",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Sistem Rekomendasi Hotel dengan Alokasi Kamar Berbasis Kelompok")
st.markdown("""
    Sistem rekomendasi hotel yang mengintegrasikan content-based filtering dengan alokasi kamar berbasis kelompok.
    Sistem ini mempertimbangkan ukuran kelompok dan preferensi spesifik untuk memberikan rekomendasi yang lebih relevan.
""")

# Preprocessing functions
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        words = text.split()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        # Stemming
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]
        # Join back
        return ' '.join(words)
    return ''

# Load data
@st.cache_data
def load_data():
    hotels_df = pd.read_csv('processed_hotel_data.csv')

    hotels_df['combined_features'] = hotels_df['name'] + ' ' + hotels_df['city'] + ' ' + hotels_df['region'] + ' ' + hotels_df['hotelFeatures']
    hotels_df['processed_features'] = hotels_df['combined_features'].apply(preprocess_text)

    return hotels_df

    # This is a placeholder for your actual data loading logic
    # In a real application, you would load your CSV file here
    # For demonstration, we'll create a small synthetic dataset
    
    # Example: hotels_df = pd.read_csv('your_hotel_dataset.csv')
    
    # Create sample data
    data = {
        'id': list(range(1, 21)),
        'name': [f"Hotel {i}" for i in range(1, 21)],
        'city': ['Jakarta', 'Surabaya', 'Bali', 'Bandung', 'Yogyakarta'] * 4,
        'region': ['West Java', 'East Java', 'Bali', 'West Java', 'Central Java'] * 4,
        'starRating': np.random.randint(1, 6, size=20),
        'userRating': np.random.uniform(2.5, 5.0, size=20).round(1),
        'numReviews': np.random.randint(10, 500, size=20),
        'latitude': np.random.uniform(-8.0, -5.0, size=20),
        'longitude': np.random.uniform(105.0, 120.0, size=20),
        'cheapestRate_perNight_totalFare': np.random.uniform(300000, 2000000, size=20),
        'hotelFeatures': [
            'Pool, Wifi, Restaurant, Spa',
            'Wifi, Restaurant, Fitness Center',
            'Beach, Pool, Restaurant, Kids Club, Spa',
            'Mountain View, Restaurant, Wifi',
            'Cultural Tours, Wifi, Restaurant',
            'Pool, Wifi, Restaurant, Business Center',
            'Wifi, Restaurant, Airport Shuttle',
            'Beach, Pool, Restaurant, Spa',
            'City View, Restaurant, Wifi, Shopping',
            'Quiet, Garden, Wifi, Restaurant',
            'Pool, Kids Club, Restaurant, Family Rooms',
            'Accessible, Elevator, Restaurant, Quiet',
            'Business Center, Wifi, Meeting Rooms',
            'Pool, Beach, Diving, Snorkeling',
            'Cultural Experience, Local Food, Wifi',
            'Shopping, City Center, Restaurant',
            'Mountain Hiking, Restaurant, Adventure',
            'Family Rooms, Kids Menu, Pool, Playground',
            'Luxury Spa, Fine Dining, Pool',
            'Budget Friendly, Basic, Clean'
        ],
        'hotelCategory': ['Luxury', 'Business', 'Family', 'Budget', 'Resort'] * 4,
    }
    
    df = pd.DataFrame(data)
    
    # Create combined features
    df['combined_features'] = df['name'] + ' ' + df['city'] + ' ' + df['region'] + ' ' + df['hotelFeatures']
    
    # Preprocess features
    df['processed_features'] = df['combined_features'].apply(preprocess_text)
    
    return df

# Load the data
hotels_df = load_data()

# Create TF-IDF matrix and cosine similarity
@st.cache_data
def create_similarity_matrix(df):
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['processed_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = create_similarity_matrix(hotels_df)

# Basic recommendation function
def get_basic_recommendations(hotel_index, num_recs=10):
    sim_scores = list(enumerate(cosine_sim[hotel_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recs+1]
    hotel_indices = [i[0] for i in sim_scores]
    return hotels_df.iloc[hotel_indices].reset_index(drop=True)

# Group-based recommendation function
def get_group_recommendations(hotel_index, group_size, num_recs=10):
    # Get basic recommendations (more for filtering)
    basic_recs = get_basic_recommendations(hotel_index, num_recs * 2)
    
    # Determine price factor based on group size
    if group_size <= 2:
        price_factor = 1.0
    elif group_size <= 4:
        price_factor = 1.5
    elif group_size <= 6:
        price_factor = 2.0
    else:
        price_factor = 2.5
    
    # Get reference hotel
    ref_hotel = hotels_df.iloc[hotel_index]
    
    # Calculate appropriate price range
    target_price = ref_hotel['cheapestRate_perNight_totalFare'] * price_factor
    min_price = target_price * 0.7
    max_price = target_price * 1.3
    
    # Filter based on price range
    filtered_recs = basic_recs[
        (basic_recs['cheapestRate_perNight_totalFare'] >= min_price) & 
        (basic_recs['cheapestRate_perNight_totalFare'] <= max_price)
    ]
    
    # If not enough recommendations after filtering, use basic recs
    if len(filtered_recs) < num_recs:
        filtered_recs = basic_recs
    
    # Sort by user rating and stars
    filtered_recs = filtered_recs.sort_values(by=['userRating', 'starRating'], ascending=False)
    
    # Get top N
    return filtered_recs.head(num_recs).reset_index(drop=True)

# Advanced group recommendation function
def get_advanced_group_recommendations(hotel_index, group_composition, preferences, num_recs=10):
    # Calculate group size
    group_size = sum(group_composition.values())
    
    # Get basic group recommendations
    basic_group_recs = get_group_recommendations(hotel_index, group_size, num_recs * 2)
    
    # Function to calculate preference score
    def calculate_preference_score(hotel_features):
        score = 0
        for pref, weight in preferences.items():
            if pref.lower() in str(hotel_features).lower():
                score += weight
        return score / len(preferences) if preferences else 0
    
    # Function to calculate demographic score
    def calculate_demographic_score(hotel_features):
        score = 0
        if group_composition.get('children', 0) > 0:
            child_keywords = ['kids', 'family', 'playground', 'children']
            child_score = 0
            for keyword in child_keywords:
                if keyword in str(hotel_features).lower():
                    child_score += 1
            score += child_score / len(child_keywords) * group_composition['children'] / group_size
            
        if group_composition.get('seniors', 0) > 0:
            senior_keywords = ['accessible', 'elevator', 'quiet', 'assistance']
            senior_score = 0
            for keyword in senior_keywords:
                if keyword in str(hotel_features).lower():
                    senior_score += 1
            score += senior_score / len(senior_keywords) * group_composition['seniors'] / group_size
            
        return score
    
    # Calculate composite scores
    basic_group_recs['preference_score'] = basic_group_recs['hotelFeatures'].apply(calculate_preference_score)
    basic_group_recs['demographic_score'] = basic_group_recs['hotelFeatures'].apply(calculate_demographic_score)
    basic_group_recs['composite_score'] = basic_group_recs['preference_score'] * 0.6 + basic_group_recs['demographic_score'] * 0.4
    
    # Sort by composite score
    advanced_recs = basic_group_recs.sort_values(by='composite_score', ascending=False)
    
    # Get top N
    return advanced_recs.head(num_recs).reset_index(drop=True)

# Sidebar for inputs
st.sidebar.header("Parameter Rekomendasi")

# Hotel selection
st.sidebar.subheader("Pilih Hotel Referensi")
reference_hotel = st.sidebar.selectbox(
    "Hotel yang Anda sukai:",
    options=hotels_df['name'].tolist()
)
hotel_index = hotels_df[hotels_df['name'] == reference_hotel].index[0]

# Recommendation type
rec_type = st.sidebar.radio(
    "Jenis Rekomendasi:",
    ["Rekomendasi Dasar", "Rekomendasi Berbasis Kelompok", "Rekomendasi Lanjutan"]
)

# Group size input for group-based recommendation
if rec_type == "Rekomendasi Berbasis Kelompok":
    group_size = st.sidebar.slider("Ukuran Kelompok:", 1, 10, 2)

# Advanced inputs for advanced recommendation
if rec_type == "Rekomendasi Lanjutan":
    st.sidebar.subheader("Komposisi Kelompok")
    adults = st.sidebar.number_input("Jumlah Dewasa:", 1, 10, 2)
    children = st.sidebar.number_input("Jumlah Anak-anak:", 0, 10, 0)
    seniors = st.sidebar.number_input("Jumlah Lansia:", 0, 10, 0)
    
    group_composition = {'adults': adults, 'children': children, 'seniors': seniors}
    
    st.sidebar.subheader("Preferensi (0-1)")
    pref_pool = st.sidebar.slider("Kolam Renang:", 0.0, 1.0, 0.5)
    pref_wifi = st.sidebar.slider("Wifi:", 0.0, 1.0, 0.8)
    pref_restaurant = st.sidebar.slider("Restoran:", 0.0, 1.0, 0.6)
    pref_spa = st.sidebar.slider("Spa:", 0.0, 1.0, 0.3)
    pref_kids = st.sidebar.slider("Fasilitas Anak:", 0.0, 1.0, 0.0 if children == 0 else 0.7)
    
    preferences = {
        'pool': pref_pool,
        'wifi': pref_wifi,
        'restaurant': pref_restaurant,
        'spa': pref_spa,
        'kids': pref_kids
    }

# Number of recommendations
num_recs = st.sidebar.slider("Jumlah Rekomendasi:", 1, 10, 5)

# Generate recommendations based on selection
if st.sidebar.button("Dapatkan Rekomendasi"):
    st.subheader(f"Hotel Referensi: {reference_hotel}")
    
    # Display reference hotel details
    ref_hotel = hotels_df.iloc[hotel_index]
    ref_col1, ref_col2, ref_col3 = st.columns(3)
    with ref_col1:
        st.write(f"**Lokasi:** {ref_hotel['city']}, {ref_hotel['region']}")
        st.write(f"**Kategori:** {ref_hotel['hotelCategory']}")
    with ref_col2:
        st.write(f"**Rating:** {'⭐' * int(ref_hotel['starRating'])}")
        st.write(f"**User Rating:** {ref_hotel['userRating']}/5 ({ref_hotel['numReviews']} reviews)")
    with ref_col3:
        st.write(f"**Harga per Malam:** Rp {int(ref_hotel['cheapestRate_perNight_totalFare']):,}")
    
    st.write(f"**Fitur Hotel:** {ref_hotel['hotelFeatures']}")
    
    st.markdown("---")
    st.subheader("Rekomendasi Hotel")
    
    # Get appropriate recommendations
    if rec_type == "Rekomendasi Dasar":
        recommendations = get_basic_recommendations(hotel_index, num_recs)
        st.info("Rekomendasi dasar berdasarkan kesamaan fitur hotel")
    elif rec_type == "Rekomendasi Berbasis Kelompok":
        recommendations = get_group_recommendations(hotel_index, group_size, num_recs)
        st.info(f"Rekomendasi untuk kelompok dengan {group_size} orang")
    else:  # Advanced
        total_people = sum(group_composition.values())
        recommendations = get_advanced_group_recommendations(hotel_index, group_composition, preferences, num_recs)
        st.info(f"Rekomendasi lanjutan untuk kelompok dengan {adults} dewasa, {children} anak-anak, dan {seniors} lansia")
    
    # Display recommendations
    for i, row in recommendations.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.subheader(row['name'])
                st.write(f"**Lokasi:** {row['city']}, {row['region']}")
                st.write(f"**Kategori:** {row['hotelCategory']}")
            with col2:
                st.write(f"**Rating:** {'⭐' * int(row['starRating'])}")
                st.write(f"**User Rating:** {row['userRating']}/5 ({row['numReviews']} reviews)")
                st.write(f"**Fitur Hotel:** {row['hotelFeatures'][:100]}...")
            with col3:
                st.write(f"**Harga per Malam:** Rp {int(row['cheapestRate_perNight_totalFare']):,}")
                
                if rec_type == "Rekomendasi Lanjutan":
                    st.write(f"**Skor Preferensi:** {row['preference_score']:.2f}")
                    st.write(f"**Skor Demografis:** {row['demographic_score']:.2f}")
                    st.write(f"**Skor Komposit:** {row['composite_score']:.2f}")
            
            st.markdown("---")

# Add an explanation section
with st.expander("Tentang Sistem Rekomendasi"):
    st.markdown("""
    ### Cara Kerja Sistem

    Sistem rekomendasi hotel ini menggunakan pendekatan **content-based filtering** yang diintegrasikan dengan pertimbangan **alokasi kamar berbasis kelompok**. Berikut cara kerjanya:
    
    #### 1. Rekomendasi Dasar
    - Menggunakan **TF-IDF** (Term Frequency-Inverse Document Frequency) untuk mengekstrak fitur dari deskripsi hotel
    - Menghitung **cosine similarity** antara hotel untuk menemukan hotel yang serupa
    
    #### 2. Rekomendasi Berbasis Kelompok
    - Memperluas rekomendasi dasar dengan mempertimbangkan **ukuran kelompok**
    - Menyesuaikan **kisaran harga** berdasarkan ukuran kelompok
    - Memfilter hotel berdasarkan kapasitas dan harga yang sesuai
    
    #### 3. Rekomendasi Lanjutan
    - Mempertimbangkan **komposisi demografis** kelompok (dewasa, anak-anak, lansia)
    - Memasukkan **preferensi spesifik** dengan bobot kepentingan
    - Menghitung skor komposit untuk memberikan rekomendasi yang paling relevan

    Sistem ini dirancang untuk mengatasi keterbatasan sistem rekomendasi tradisional yang biasanya hanya fokus pada preferensi individu.
    """)

# Footer
st.markdown("---")
st.markdown("Sistem Rekomendasi Hotel dengan Alokasi Kamar Berbasis Kelompok © 2025")