🎬 CineMate-AI
An AI-powered movie recommendation system with chatbot features and interactive quizzes to help you discover movies you'll love.

To see the demo - https://cinemate-ai-6rmbvk3pwvtsfwzitarefx.streamlit.app/


---

## 🚀 Key Features

- **NLP‑powered Conversational Interface**: Understands user prompts like “Recommend me a sci‑fi thriller” or “Who directed Inception?”
- **Context Retention**: Keeps track of session flow ("What else did they do?"—it remembers the actor, director, etc.).
- **Movie Metadata Access**: Fetches real-time information (plot summaries, cast, ratings) via external APIs.
- **Recommendation Engine**: Suggests films based on user preferences using content and/or collaborative filtering models.

---

## 🧠 Architecture & Tech Stack

### Components

1. **Intent Recognition**
   - Built using [huggingface transformers] or spaCy — fine‑tuned on movie‑related prompts.

2. **Entity Extraction**
   - Identifies movie titles, genres, actor names, etc., using NER models.

3. **Dialogue Manager**
   - Rules or lightweight RNN-based context management for maintaining conversation flow.

4. **Back-end Engines**
   - **Info Retriever**: Queries IMDB/TMDB/OMDB APIs for metadata
   - **Recommendation Engine**: Leverages content-based similarity or collaborative filtering

5. **Front-end Interface**
   - Console chatbot or integrated via Flask/Streamlit/API

---

## 🛠️ Installation & Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/abhinav‑kommalapati/CineMate‑Ai.git
   cd CineMate‑Ai
