import streamlit as st
import numpy as np
import pandas as pd
import difflib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

# OMDb API Key (from user)
OMDB_API_KEY = '56e3263d'
OMDB_API_URL = 'http://www.omdbapi.com/'

# --- Custom CSS for background and card style ---
st.markdown(
    """
    <style>
    body {
        background-color: #f5f6fa;
    }
    .stApp {
        background: linear-gradient(120deg, #f5f6fa 0%, #dff9fb 100%);
    }
    .recommendation-card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.08);
        padding: 1.2em 1.5em;
        margin-bottom: 1em;
        font-size: 1.1em;
        display: flex;
        align-items: flex-start;
    }
    .movie-poster {
        width: 100px;
        height: 150px;
        object-fit: cover;
        border-radius: 8px;
        margin-right: 1.5em;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.10);
    }
    .movie-title {
        font-weight: 600;
        color: #30336b;
        font-size: 1.2em;
    }
    .movie-info {
        color: #535c68;
        font-size: 1em;
        margin-bottom: 0.3em;
    }
    .movie-rating {
        color: #e17055;
        font-weight: 600;
        font-size: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Title and Description ---
st.markdown("<h1 style='text-align:center; color:#30336b;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#535c68;'>Find your next favorite movie based on what you love!</h4>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#686de0; margin-bottom:2em;'>
    Enter a movie name below and get personalized recommendations instantly.
</div>
""", unsafe_allow_html=True)

# --- Load data ---
@st.cache_data

def load_data():
    return pd.read_csv('movies.csv')

movies_data = load_data()

# Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

# Vectorize
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity
similarity = cosine_similarity(feature_vectors)

# --- Autocomplete for movie search (show suggestions only) ---
def autocomplete_movie(query, titles, limit=10):
    query = query.lower()
    return [title for title in titles if query in title.lower()][:limit]

# --- Centered Input ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    all_titles = movies_data['title'].tolist()
    search_query = st.text_input('Enter your favourite movie name:', key='movie_input')
    suggestions = autocomplete_movie(search_query, all_titles) if search_query else []
    if suggestions and search_query:
        st.markdown('<div style="color:#686de0; margin-bottom:0.5em;">Suggestions:</div>', unsafe_allow_html=True)
        for s in suggestions:
            st.markdown(f'- {s}')

if search_query:
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(search_query, list_of_all_titles)
    if find_close_match:
        with col2:
            close_match = st.selectbox('Did you mean:', find_close_match)
            recommend_btn = st.button('Show Recommendations', use_container_width=True)
        if recommend_btn:
            with st.spinner('Finding your next favorite movies...'):
                index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
                similarity_score = list(enumerate(similarity[index_of_the_movie]))
                sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
                st.markdown("<h3 style='color:#30336b;'>Top 10 Recommended Movies:</h3>", unsafe_allow_html=True)
                for idx, i in enumerate(sorted_similar_movies[1:11]):
                    movie_row = movies_data[movies_data.index == i[0]].iloc[0]
                    movie_title = movie_row['title']
                    year = movie_row.get('release_date', '')[:4]
                    genres = movie_row.get('genres', '')
                    tagline = movie_row.get('tagline', '')
                    director = movie_row.get('director', '')
                    cast = movie_row.get('cast', '').split(' ')[:5]
                    # Fetch OMDb info
                    params = {'t': movie_title, 'apikey': OMDB_API_KEY}
                    try:
                        resp = requests.get(OMDB_API_URL, params=params, timeout=5)
                        data = resp.json() if resp.status_code == 200 else {}
                    except Exception:
                        data = {}
                    poster_url = data.get('Poster', '') if data.get('Poster', '') != 'N/A' else ''
                    imdb_rating = data.get('imdbRating', 'N/A')
                    plot = data.get('Plot', tagline)
                    # Card layout
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <div>
                            {'<img src="'+poster_url+'" class="movie-poster">' if poster_url else ''}
                        </div>
                        <div>
                            <div class='movie-title'>{idx+1}. {movie_title} ({year})</div>
                            <div class='movie-info'><b>Genres:</b> {genres}</div>
                            <div class='movie-info'><b>Director:</b> {director}</div>
                            <div class='movie-info'><b>Main Cast:</b> {' '.join(cast)}</div>
                            <div class='movie-info'><b>Plot:</b> {plot}</div>
                            <div class='movie-rating'>IMDb: {imdb_rating}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning('No close matches found. Please check your input.')
else:
    st.markdown("<div style='color:#000000; font-size:1.1em; background:#f5f6fa; padding:0.7em 1em; border-radius:8px; border:1px solid #dfe4ea; text-align:center;'>Type a movie name to get recommendations!</div>", unsafe_allow_html=True)

# --- Chatbot and Quiz Mode in Sidebar ---
st.sidebar.title('🎬 Movie Chatbot & Quiz')
st.sidebar.markdown('Ask me anything about movies, or try the Movie Quiz!')

quiz_mode = st.sidebar.checkbox('Quiz Mode 🎲', key='quiz_mode')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'quiz_history' not in st.session_state:
    st.session_state['quiz_history'] = []
if 'quiz_question' not in st.session_state:
    st.session_state['quiz_question'] = ''
if 'quiz_answer' not in st.session_state:
    st.session_state['quiz_answer'] = ''
if 'awaiting_quiz_answer' not in st.session_state:
    st.session_state['awaiting_quiz_answer'] = False

api_url = 'https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta'
headers = {
    'Authorization': f"Bearer {st.secrets['huggingface']['api_token']}",
    'Content-Type': 'application/json'
}

if quiz_mode:
    st.sidebar.markdown('**Quiz Mode is ON!** The AI will ask you movie trivia questions. Try to answer as many as you can!')
    if not st.session_state['awaiting_quiz_answer']:
        # Ask the LLM to generate a trivia question
        prompt = "Ask me a single, fun, and challenging movie trivia question. Only output the question, nothing else."
        data = {"inputs": prompt}
        try:
            with st.spinner('Getting a trivia question...'):
                response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and 'generated_text' in result[0]:
                        raw_output = result[0]['generated_text'].strip()
                    else:
                        raw_output = result.get('generated_text', '').strip()
                    # Extract only the first question
                    match = re.search(r'([^\n\.!?]*\?)', raw_output)
                    if match:
                        question = match.group(1).strip()
                    else:
                        question = "Which movie won the Oscar for Best Picture in 1994?"
                    st.session_state['quiz_question'] = question
                    st.session_state['awaiting_quiz_answer'] = True
                else:
                    st.sidebar.error('Could not get a trivia question.')
        except Exception as e:
            st.sidebar.error(f'Error: {e}')
    if st.session_state['quiz_question']:
        st.sidebar.markdown(f"**Question:** {st.session_state['quiz_question']}")
        user_quiz_answer = st.sidebar.text_input('Your Answer:', key='quiz_input')
        if st.sidebar.button('Submit Answer', key='submit_quiz') and user_quiz_answer.strip():
            # Ask the LLM to check the answer
            check_prompt = f"Question: {st.session_state['quiz_question']}\nUser Answer: {user_quiz_answer}\nIs this correct? If not, give the correct answer."
            data = {"inputs": check_prompt}
            try:
                with st.spinner('Checking your answer...'):
                    response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and 'generated_text' in result[0]:
                            feedback = result[0]['generated_text'].strip()
                        else:
                            feedback = result.get('generated_text', '').strip()
                        st.session_state['quiz_history'].append({'question': st.session_state['quiz_question'], 'user_answer': user_quiz_answer, 'feedback': feedback})
                        st.session_state['quiz_question'] = ''
                        st.session_state['awaiting_quiz_answer'] = False
                    else:
                        st.sidebar.error('Could not check your answer.')
            except Exception as e:
                st.sidebar.error(f'Error: {e}')
    # Show quiz history
    if st.session_state['quiz_history']:
        st.sidebar.markdown('---')
        st.sidebar.markdown('**Quiz History:**')
        for q in st.session_state['quiz_history'][-5:][::-1]:
            st.sidebar.markdown(f"- **Q:** {q['question']}\n- **Your Answer:** {q['user_answer']}\n- **AI Feedback:** {q['feedback']}")
else:
    # Normal chatbot mode
    user_input = st.sidebar.text_input('You:', key='chat_input')
    if st.sidebar.button('Send', key='send_btn') and user_input.strip():
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        prompt = "\n".join([f"User: {msg['content']}" if msg['role']=='user' else f"Assistant: {msg['content']}" for msg in st.session_state['chat_history']])
        data = {"inputs": prompt + "\nAssistant:"}
        try:
            with st.spinner('Thinking...'):
                response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and 'generated_text' in result[0]:
                        answer = result[0]['generated_text'].split('Assistant:')[-1].strip()
                    else:
                        answer = result.get('generated_text', '').split('Assistant:')[-1].strip()
                    st.session_state['chat_history'].append({'role': 'assistant', 'content': answer})
                else:
                    try:
                        error_detail = response.json()
                    except Exception:
                        error_detail = response.text
                    st.session_state['chat_history'].append({'role': 'assistant', 'content': f"[Error {response.status_code}] {error_detail}"})
        except Exception as e:
            st.session_state['chat_history'].append({'role': 'assistant', 'content': f'Error: {e}'})
    # Display chat history
    for msg in st.session_state['chat_history'][-8:]:
        if msg['role'] == 'user':
            st.sidebar.markdown(f"**You:** {msg['content']}")
        else:
            st.sidebar.markdown(f"**AI:** {msg['content']}") 