import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Data Sosial Media",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Analytics Dashboard")

@st.cache_data
def load_data():
    """Memuat dan memproses data"""
    try:
        # Load data
        users = pd.read_csv('https://raw.githubusercontent.com/cayaaa14/data-real/refs/heads/main/user_table.csv')
        friends = pd.read_csv('https://raw.githubusercontent.com/cayaaa14/data-real/refs/heads/main/friends_table.csv')
        posts = pd.read_csv('https://raw.githubusercontent.com/cayaaa14/data-real/refs/heads/main/posts_table.csv')
        reactions = pd.read_csv('https://raw.githubusercontent.com/cayaaa14/data-real/refs/heads/main/reactions_table.csv')
        
        # Data cleaning
        users['Subscription Date'] = pd.to_datetime(users['Subscription Date'], unit='s')
        posts['Post Date'] = pd.to_datetime(posts['Post Date'], unit='s')
        
        # Clean reactions data
        reactions = reactions.dropna(subset=['User'])
        mode_rt = reactions['Reaction Type'].mode()[0]
        reactions['Reaction Type'] = reactions['Reaction Type'].fillna(mode_rt)
        median_rd = reactions['Reaction Date'].median()
        reactions['Reaction Date'] = reactions['Reaction Date'].fillna(median_rd)
        reactions['Reaction Date'] = pd.to_datetime(reactions['Reaction Date'], unit='s')
        
        # Remove duplicates
        users = users.drop_duplicates()
        friends = friends.drop_duplicates()
        posts = posts.drop_duplicates()
        reactions = reactions.drop_duplicates()
        
        return users, friends, posts, reactions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_data
def create_integrated_dataset(users, friends, posts, reactions):
    """Membuat dataset terintegrasi"""
    # Base dari users
    users = users.copy()
    users['user_id'] = range(1, len(users) + 1)
    integrated_data = users[['user_id', 'Name', 'Surname', 'Age', 'Subscription Date']].copy()
    
    # Data teman
    all_friends = pd.concat([
        friends[['Friend 1']].rename(columns={'Friend 1': 'user_id'}),
        friends[['Friend 2']].rename(columns={'Friend 2': 'user_id'})
    ])
    friend_stats = all_friends.value_counts('user_id').reset_index()
    friend_stats.columns = ['user_id', 'friend_count']
    
    # Data posting
    posts = posts.copy()
    posts = posts.rename(columns={'User': 'user_id'})
    posts['post_id'] = range(1, len(posts) + 1)
    post_stats = posts.groupby('user_id').agg({'post_id': 'count'}).reset_index()
    post_stats.columns = ['user_id', 'post_count']
    
    # Data reaksi diberikan
    reactions = reactions.copy()
    reactions = reactions.rename(columns={'User': 'user_id'})
    reactions['reaction_id'] = range(1, len(reactions) + 1)
    reactions_given = reactions.groupby('user_id').agg({'reaction_id': 'count'}).reset_index()
    reactions_given.columns = ['user_id', 'reactions_given']
    
    # Simulasi reaksi terhadap posting
    reactions['post_id'] = reactions['reaction_id'] % len(posts) + 1
    post_reactions = posts.merge(reactions, on='post_id', how='inner')
    reactions_received = post_reactions.groupby('user_id_x').agg({'user_id_y': 'count'}).reset_index()
    reactions_received.columns = ['user_id', 'reactions_received']
    
    # Gabungkan semua data
    integrated_data = integrated_data.merge(friend_stats, on='user_id', how='left')
    integrated_data = integrated_data.merge(post_stats, on='user_id', how='left')
    integrated_data = integrated_data.merge(reactions_given, on='user_id', how='left')
    integrated_data = integrated_data.merge(reactions_received, on='user_id', how='left')
    
    # Bersihkan missing values
    integrated_data.fillna(0, inplace=True)
    
    # Tambahan kolom analisis
    integrated_data['age_group'] = pd.cut(
        integrated_data['Age'],
        bins=[0, 20, 30, 40, 50, 100],
        labels=['<20', '20-29', '30-39', '40-49', '50+']
    )
    integrated_data['registration_year'] = pd.to_datetime(integrated_data['Subscription Date']).dt.year
    integrated_data['is_active_poster'] = integrated_data['post_count'] > 0
    integrated_data['is_social'] = integrated_data['friend_count'] > 0
    integrated_data['engagement_ratio'] = integrated_data['reactions_received'] / (integrated_data['post_count'] + 1)
    integrated_data['total_activity'] = (integrated_data['friend_count'] + 
                                       integrated_data['post_count'] + 
                                       integrated_data['reactions_given'])
    
    return integrated_data, posts, reactions

users, friends, posts, reactions = load_data()
integrated_data, posts_processed, reactions_processed = create_integrated_dataset(users, friends, posts, reactions)

st.sidebar.header("🧭 Insight Navigator")
selected_insight = st.sidebar.radio("Pilih pertanyaan", [
    "1. Total Pengguna",
    "2. Distribusi Usia Pengguna",
    "3. Top Content Creator",
    "4. Jenis Reaksi Populer",
    "5. Aktivitas Pengguna per Kelompok Usia",
    "6. Hubungan Jumlah Teman dan Postingan",
    "7. Pengguna Paling Aktif",
    "8. Timeline Aktivitas Reaksi",
    "9. Level Aktivitas Pengguna",
    "10. Matriks Korelasi"
])

# 1. Total Pengguna
if selected_insight == "1. Total Pengguna":
    st.subheader("👥 Total Pengguna")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", f"{len(users):,}")
    with col2:
        st.metric("Total Friendships", f"{len(friends):,}")
    with col3:
        st.metric("Total Posts", f"{len(posts):,}")
    with col4:
        st.metric("Total Reactions", f"{len(reactions):,}")

# 2. Distribusi Usia Pengguna
elif selected_insight == "2. Distribusi Usia Pengguna":
    st.subheader("📊 Distribusi Usia Pengguna")
    fig = px.histogram(
        integrated_data,
        x='Age',
        title="Distribusi Usia Pengguna",
        nbins=30
    )
    st.plotly_chart(fig, use_container_width=True)

# 3. Top Content Creator
elif selected_insight == "3. Top Content Creator":
    st.subheader("🏆 Top Content Creator")
    top_n = st.slider("Jumlah pengguna yang ditampilkan:", 5, 20, 10)
    top_posters = integrated_data.nlargest(top_n, 'post_count')
    fig = px.bar(
        top_posters,
        x='post_count',
        y=[f"{row['Name']} {row['Surname']}" for _, row in top_posters.iterrows()],
        orientation='h',
        title=f"Top {top_n} Content Creator"
    )
    st.plotly_chart(fig, use_container_width=True)

# 4. Jenis Reaksi Populer
elif selected_insight == "4. Jenis Reaksi Populer":
    st.subheader("💝 Jenis Reaksi Populer")
    reaction_counts = reactions_processed['Reaction Type'].value_counts()
    fig = px.pie(
        values=reaction_counts.values,
        names=reaction_counts.index,
        title="Proporsi Jenis Reaksi"
    )
    st.plotly_chart(fig, use_container_width=True)

# 5. Aktivitas Pengguna per Kelompok Usia
elif selected_insight == "5. Aktivitas Pengguna per Kelompok Usia":
    st.subheader("📈 Aktivitas Pengguna per Kelompok Usia")
    age_activity = integrated_data.groupby('age_group')[['friend_count', 'post_count', 'reactions_given']].mean()
    fig = px.line(
        age_activity,
        title="Rata-rata Aktivitas per Kelompok Usia",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# 6. Hubungan Jumlah Teman dan Postingan
elif selected_insight == "6. Hubungan Jumlah Teman dan Postingan":
    st.subheader("🔗 Hubungan Jumlah Teman dan Postingan")
    fig = px.scatter(
        integrated_data,
        x='friend_count',
        y='post_count',
        color='age_group',
        title="Jumlah Teman vs Jumlah Postingan"
    )
    st.plotly_chart(fig, use_container_width=True)

# 7. Pengguna Paling Aktif
elif selected_insight == "7. Pengguna Paling Aktif":
    st.subheader("🏅 Pengguna Paling Aktif")
    top_n = st.slider("Jumlah pengguna yang ditampilkan:", 5, 20, 10)
    top_active = integrated_data.nlargest(top_n, 'total_activity')
    fig = px.bar(
        top_active,
        x='total_activity',
        y=[f"{row['Name']} {row['Surname']}" for _, row in top_active.iterrows()],
        orientation='h',
        title=f"Top {top_n} Pengguna Paling Aktif"
    )
    st.plotly_chart(fig, use_container_width=True)

# 8. Timeline Aktivitas Reaksi
elif selected_insight == "8. Timeline Aktivitas Reaksi":
    st.subheader("⏳ Timeline Aktivitas Reaksi")
    reactions_timeline = reactions_processed.copy()
    reactions_timeline['date'] = pd.to_datetime(reactions_timeline['Reaction Date']).dt.date
    daily_reactions = reactions_timeline.groupby('date').size().reset_index(name='count')
    fig = px.line(
        daily_reactions,
        x='date',
        y='count',
        title="Aktivitas Reaksi Harian"
    )
    st.plotly_chart(fig, use_container_width=True)

# 9. Level Aktivitas Pengguna
elif selected_insight == "9. Level Aktivitas Pengguna":
    st.subheader("📶 Level Aktivitas Pengguna")
    def categorize_activity(row):
        if row['total_activity'] == 0:
            return 'Tidak Aktif'
        elif row['total_activity'] <= 5:
            return 'Aktivitas Rendah'
        elif row['total_activity'] <= 15:
            return 'Aktivitas Sedang'
        elif row['total_activity'] <= 30:
            return 'Aktivitas Tinggi'
        else:
            return 'Sangat Aktif'
    
    integrated_data['activity_level'] = integrated_data.apply(categorize_activity, axis=1)
    activity_counts = integrated_data['activity_level'].value_counts()
    fig = px.pie(
        values=activity_counts.values,
        names=activity_counts.index,
        title="Distribusi Level Aktivitas Pengguna"
    )
    st.plotly_chart(fig, use_container_width=True)

# 10. Matriks Korelasi
elif selected_insight == "10. Matriks Korelasi":
    st.subheader("🔍 Matriks Korelasi")
    correlation_vars = ['Age', 'friend_count', 'post_count', 'reactions_given', 'reactions_received', 'total_activity']
    correlation_matrix = integrated_data[correlation_vars].corr()
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Korelasi Antar Variabel"
    )
    st.plotly_chart(fig, use_container_width=True)