import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Set page config
st.set_page_config(page_title="Movies and Tv Series Recommender", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('preprocessed_netflix_data.csv')
    return df

@st.cache_data
def create_similarity_matrix(df):
    # Combine relevant features
    df['combined_features'] = (df['description'].fillna('') + ' ' + 
                              df['listed_in'].fillna('') + ' ' + 
                              df['cast'].fillna('') + ' ' + 
                              df['director'].fillna(''))
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

def get_recommendations(title, df, similarity_matrix):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'description', 'type', 'release_year', 'rating', 'duration', 'listed_in']]

def main():
    st.title("🎬 Movies and Tv Series Recommender")
    
    # Load data
    df = load_data()
    similarity_matrix = create_similarity_matrix(df)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    content_type = st.sidebar.multiselect("Select Content Type", 
                                          options=df['type'].unique(),
                                          default=df['type'].unique())
    
    # Filter dataframe based on content type
    filtered_df = df[df['type'].isin(content_type)]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Find Similar Content")
        selected_title = st.selectbox("Select a show or movie:", 
                                      filtered_df['title'].values)
        
        if st.button('Get Recommendations'):
            recommendations = get_recommendations(selected_title, df, similarity_matrix)
            
            # Display original content info
            original_content = df[df['title'] == selected_title].iloc[0]
            st.write("### Selected Content")
            st.write(f"**{original_content['title']}** ({original_content['type']})")
            st.write(f"**Release Year:** {original_content['release_year']}")
            st.write(f"**Rating:** {original_content['rating']}")
            st.write(f"**Duration:** {original_content['duration']}")
            st.write(f"**Categories:** {original_content['listed_in']}")
            st.write(f"**Description:** {original_content['description']}")
            
            # Display recommendations
            st.write("### Recommended Content")
            for _, row in recommendations.iterrows():
                with st.expander(f"{row['title']} ({row['type']})"):
                    st.write(f"**Release Year:** {row['release_year']}")
                    st.write(f"**Rating:** {row['rating']}")
                    st.write(f"**Duration:** {row['duration']}")
                    st.write(f"**Categories:** {row['listed_in']}")
                    st.write(f"**Description:** {row['description']}")
    
    with col2:
        st.subheader("Content Statistics")
        
        # Content type distribution
        type_counts = filtered_df['type'].value_counts()
        fig_type = px.pie(values=type_counts.values, 
                          names=type_counts.index, 
                          title="Content Type Distribution")
        st.plotly_chart(fig_type)
        
        # Rating distribution
        rating_counts = filtered_df['rating'].value_counts()
        fig_rating = px.bar(x=rating_counts.index, 
                            y=rating_counts.values,
                            title="Rating Distribution")
        st.plotly_chart(fig_rating)

if __name__ == "__main__":
    main()