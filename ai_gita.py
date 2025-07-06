import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions
import plotly.graph_objects as go
import networkx as nx
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Optional
import textwrap
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure logger
logging.basicConfig(
    filename='gita_ai.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Constants
DATA_DIR = "gita_dir"  # Directory containing 18 JSON files
COMMENTATORS = [
    "Swami Ramsukhdas", "Sri Anandgiri", "Sri Dhanpati", 
    "Sri Neelkanth", "Sri Ramanuja", "Sri Sridhara Swami",
    "Swami Chinmayananda", "Swami Sivananda"
]
TRANSLATORS = [
    "swami ramsukhdas", "swami tejomayananda", 
    "swami sivananda", "shri purohit swami"
]

# Initialize ChromaDB
chroma_client = chromadb.Client()
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def validate_verse_structure(verse: Dict) -> bool:
    """Validate the structure and content of a verse dictionary"""
    required_keys = {'chapter', 'verse', 'text', 'commentaries', 'translations'}
    if not all(key in verse for key in required_keys):
        logging.warning(f"Missing required keys in verse: {verse.get('chapter', 'unknown')}.{verse.get('verse', 'unknown')}")
        return False
    
    if not isinstance(verse['text'], str) or not verse['text'].strip():
        logging.warning(f"Invalid text in verse: {verse['chapter']}.{verse['verse']}")
        return False
    
    if not isinstance(verse['commentaries'], dict):
        logging.warning(f"Invalid commentaries in verse: {verse['chapter']}.{verse['verse']}")
        return False
        
    if not isinstance(verse['translations'], dict):
        logging.warning(f"Invalid translations in verse: {verse['chapter']}.{verse['verse']}")
        return False
        
    return True

def clean_text(text: Optional[str]) -> str:
    """Clean and validate text input"""
    if text is None:
        return ""
    return str(text).strip()

def process_commentaries(commentaries: Dict, verse_id: str) -> List[Tuple]:
    """Process and validate commentaries for a verse"""
    valid_commentaries = []
    for commentator, commentary in commentaries.items():
        if commentator in COMMENTATORS:
            clean_comm = clean_text(commentary)
            if clean_comm:
                comm_id = f"{verse_id}_{commentator}"
                metadata = {
                    'verse_id': verse_id,
                    'commentator': commentator,
                    'chapter': int(verse_id.split('.')[0]),
                    'verse': int(verse_id.split('.')[1])
                }
                valid_commentaries.append((clean_comm, metadata, comm_id))
            else:
                logging.warning(f"Empty commentary from {commentator} for verse {verse_id}")
    return valid_commentaries

def process_translations(translations: Dict, verse_id: str) -> List[Tuple]:
    """Process and validate translations for a verse"""
    valid_translations = []
    for translator, translation in translations.items():
        if translator.lower() in [t.lower() for t in TRANSLATORS]:
            clean_trans = clean_text(translation)
            if clean_trans:
                trans_id = f"{verse_id}_{translator}"
                valid_translations.append((clean_trans, translator))
            else:
                logging.warning(f"Empty translation from {translator} for verse {verse_id}")
    return valid_translations

class GitaAICounselor:
    def __init__(self):
        self.verses = []
        self.commentaries = {}
        self.translations = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ollama_endpoint = "http://localhost:11434/api/generate"
        
        # Initialize collections
        self._init_collections()
        self._load_data()
        
    def _init_collections(self):
        """Initialize ChromaDB collections with error handling"""
        try:
            if "gita_verses" not in chroma_client.list_collections():
                self.verse_collection = chroma_client.create_collection(
                    name="gita_verses",
                    embedding_function=embedding_func
                )
            else:
                self.verse_collection = chroma_client.get_collection("gita_verses")
                
            if "gita_commentaries" not in chroma_client.list_collections():
                self.commentary_collection = chroma_client.create_collection(
                    name="gita_commentaries",
                    embedding_function=embedding_func
                )
            else:
                self.commentary_collection = chroma_client.get_collection("gita_commentaries")
                
        except Exception as e:
            logging.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def _load_data(self):
        """Load all Gita data from JSON files with parallel processing"""
        logging.info("Loading Gita data...")
        try:
            files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
            
            with ThreadPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(self._process_file, files),
                    total=len(files),
                    desc="Processing Gita files"
                ))
            
            # Flatten results and build verse index
            verses = []
            for chapter_verses in results:
                if chapter_verses:  # Skip empty results
                    verses.extend(chapter_verses)
            
            self.verses = verses
            logging.info(f"Successfully loaded {len(self.verses)} verses with commentaries")
            
        except Exception as e:
            logging.error(f"Error loading Gita data: {str(e)}")
            raise

    def _process_file(self, filename: str) -> List[Dict]:
        """Process a single JSON file and return valid verses"""
        try:
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'BhagavadGitaChapter' not in data:
                logging.warning(f"Invalid structure in file {filename}")
                return []
                
            valid_verses = []
            for verse in data['BhagavadGitaChapter']:
                if not validate_verse_structure(verse):
                    continue
                    
                verse_id = f"{verse['chapter']}.{verse['verse']}"
                verse_text = clean_text(verse['text'])
                
                if not verse_text:
                    logging.warning(f"Empty verse text for {verse_id} in {filename}")
                    continue
                
                # Add verse to collection
                try:
                    self.verse_collection.add(
                        documents=[verse_text],
                        metadatas=[{
                            'chapter': verse['chapter'],
                            'verse': verse['verse'],
                            'verse_id': verse_id
                        }],
                        ids=[verse_id]
                    )
                except Exception as e:
                    logging.error(f"Error adding verse {verse_id}: {str(e)}")
                    continue
                
                # Process commentaries
                comm_data = process_commentaries(verse['commentaries'], verse_id)
                if comm_data:
                    try:
                        self.commentary_collection.add(
                            documents=[c[0] for c in comm_data],
                            metadatas=[c[1] for c in comm_data],
                            ids=[c[2] for c in comm_data]
                        )
                    except Exception as e:
                        logging.error(f"Error adding commentaries for {verse_id}: {str(e)}")
                
                # Process translations
                trans_data = process_translations(verse['translations'], verse_id)
                
                # Build verse object
                verse_obj = {
                    'id': verse_id,
                    'chapter': verse['chapter'],
                    'verse': verse['verse'],
                    'text': verse_text,
                    'commentaries': {c[1]['commentator']: c[0] for c in comm_data},
                    'translations': {t[1]: t[0] for t in trans_data}
                }
                valid_verses.append(verse_obj)
            
            return valid_verses
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in file {filename}: {str(e)}")
            return []
        except Exception as e:
            logging.error(f"Error processing file {filename}: {str(e)}")
            return []

    async def find_relevant_verses(self, query: str, n_results: int = 5) -> List[Dict]:
        """Find verses relevant to user's query"""
        try:
            # First search in verses
            verse_results = self.verse_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Then search in commentaries
            commentary_results = self.commentary_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Combine and deduplicate results
            combined = []
            seen_verse_ids = set()
            
            for i in range(len(verse_results['ids'][0])):
                verse_id = verse_results['ids'][0][i]
                if verse_id not in seen_verse_ids:
                    verse = self.get_verse_details(verse_id)
                    if verse:
                        combined.append({
                            'id': verse_id,
                            'text': verse_results['documents'][0][i],
                            'metadata': verse_results['metadatas'][0][i],
                            'score': verse_results['distances'][0][i],
                            'type': 'verse'
                        })
                        seen_verse_ids.add(verse_id)
                    
            for i in range(len(commentary_results['ids'][0])):
                verse_id = commentary_results['metadatas'][0][i]['verse_id']
                if verse_id not in seen_verse_ids:
                    verse = self.get_verse_details(verse_id)
                    if verse:
                        combined.append({
                            'id': verse_id,
                            'text': verse['text'],
                            'metadata': commentary_results['metadatas'][0][i],
                            'score': commentary_results['distances'][0][i],
                            'type': 'commentary',
                            'commentator': commentary_results['metadatas'][0][i]['commentator']
                        })
                        seen_verse_ids.add(verse_id)
            
            # Sort by relevance score
            combined.sort(key=lambda x: x['score'])
            
            return combined[:n_results]
            
        except Exception as e:
            logging.error(f"Error finding relevant verses: {str(e)}")
            return []

    async def generate_ai_response(self, prompt: str, model: str = "mistral") -> str:
        """Generate response using Ollama"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 2000
                        }
                    }
                    
                    async with session.post(
                        self.ollama_endpoint,
                        json=payload,
                        timeout=60
                    ) as resp:
                        response = await resp.json()
                        return response.get('response', '').strip()
            
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    logging.warning("Ollama request timed out")
                    return "Analysis timed out after multiple attempts"
                await asyncio.sleep(retry_delay)
            
            except Exception as e:
                logging.error(f"Ollama error (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return f"Analysis error: {str(e)}"
                await asyncio.sleep(retry_delay)

    def get_verse_details(self, verse_id: str) -> Optional[Dict]:
        """Get full details for a specific verse"""
        for verse in self.verses:
            if verse['id'] == verse_id:
                return verse
        logging.warning(f"Verse not found: {verse_id}")
        return None

    def get_commentaries(self, verse_id: str) -> Dict[str, str]:
        """Get all commentaries for a verse"""
        verse = self.get_verse_details(verse_id)
        return verse.get('commentaries', {}) if verse else {}

    def get_translations(self, verse_id: str) -> Dict[str, str]:
        """Get all translations for a verse"""
        verse = self.get_verse_details(verse_id)
        return verse.get('translations', {}) if verse else {}

    def generate_concept_graph(self, verse_id: str) -> Optional[go.Figure]:
        """Generate network graph of related concepts"""
        verse = self.get_verse_details(verse_id)
        if not verse:
            return None
            
        # Extract key concepts using AI (simplified for demo)
        concepts = ["Dharma", "Karma", "Self", "Action", "Wisdom"]  # Would use LLM in production
        
        # Build graph
        G = nx.Graph()
        G.add_node(verse_id, label=f"BG {verse_id}", size=20)
        
        for concept in concepts:
            G.add_node(concept, label=concept, size=15)
            G.add_edge(verse_id, concept, weight=1)
            
        # Layout
        pos = nx.spring_layout(G)
        
        # Create Plotly figure
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
            
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=[G.nodes[n]['size'] for n in G.nodes()],
                color=['lightblue' if n == verse_id else 'lightgreen' for n in G.nodes()],
                line=dict(width=2, color='DarkSlateGrey')
            ))
            
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f"Concept Map for BG {verse_id}",
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                      )
        return fig

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Bhagavad Gita AI Counselor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize counselor
    if 'counselor' not in st.session_state:
        with st.spinner("Loading Bhagavad Gita AI. This may take a few minutes..."):
            st.session_state.counselor = GitaAICounselor()
    
    counselor = st.session_state.counselor
    
    st.title("🕉️ Bhagavad Gita AI Counselor")
    st.markdown("""
    <style>
    .verse-text { font-size: 18px; font-weight: bold; }
    .commentary { background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; color: black !important;}
    .translation { font-style: italic; color: #555; }
    </style>
    """, unsafe_allow_html=True)
    
    # Problem input
    with st.container():
        st.subheader("Share Your Life Challenge")
        user_problem = st.text_area(
            "Describe your current life situation or challenge:",
            placeholder="I'm struggling with...",
            height=150
        )
        
        if st.button("Find Guidance in the Gita"):
            if user_problem.strip():
                with st.spinner("Searching the Bhagavad Gita for wisdom..."):
                    relevant_verses = asyncio.run(counselor.find_relevant_verses(user_problem))
                    
                    if relevant_verses:
                        st.session_state.selected_verse = relevant_verses[0]['id']
                        st.session_state.relevant_verses = relevant_verses
                        st.success("Found relevant verses from the Gita")
                    else:
                        st.warning("No relevant verses found. Try rephrasing your challenge.")
            else:
                st.warning("Please describe your challenge to receive guidance")
    
    # Display results
    if 'selected_verse' in st.session_state:
        verse_id = st.session_state.selected_verse
        verse = counselor.get_verse_details(verse_id)
        
        if not verse:
            st.error("Verse details not found. Please try another search.")
            return
        
        st.divider()
        st.subheader(f"Bhagavad Gita {verse_id}")
        
        # Verse display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f'<div class="verse-text">{verse["text"]}</div>', unsafe_allow_html=True)
            
            # Translations
            st.subheader("Translations")
            translator = st.selectbox("Select translation", TRANSLATORS)
            if translator.lower() in [t.lower() for t in verse['translations'].keys()]:
                st.markdown(f'<div class="translation">{verse["translations"][translator]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.warning("Translation not available for selected translator")
        
        with col2:
            # Concept graph
            st.subheader("Concept Map")
            concept_fig = counselor.generate_concept_graph(verse_id)
            if concept_fig:
                st.plotly_chart(concept_fig, use_container_width=True)
            else:
                st.warning("Could not generate concept map for this verse")
        
        # Commentaries
        st.subheader("Commentaries")
        available_commentators = [c for c in COMMENTATORS if c in verse['commentaries']]
        if not available_commentators:
            st.warning("No commentaries available for this verse")
        else:
            selected_commentators = st.multiselect(
                "Select commentators to compare", 
                available_commentators,
                default=available_commentators[:2] if available_commentators else []
            )
            
            if selected_commentators:
                tabs = st.tabs(selected_commentators)
                for i, commentator in enumerate(selected_commentators):
                    with tabs[i]:
                        commentary = verse['commentaries'][commentator]
                        st.markdown(f'<div class="commentary">{commentary}</div>', unsafe_allow_html=True)
                        
                        # Debate simulator
                        if st.button(f"Ask {commentator} about this verse", key=f"debate_{i}"):
                            with st.spinner(f"Asking {commentator}..."):
                                prompt = f"""You are {commentator}, a renowned commentator on the Bhagavad Gita.
                                Verse: {verse['text']}
                                Commentary: {commentary}
                                
                                A seeker asks: "{user_problem}"
                                
                                How would you explain how this verse addresses their concern?"""
                                
                                response = asyncio.run(counselor.generate_ai_response(prompt))
                                st.markdown(f"**{commentator}:** {response}")
        
        # AI Analysis Section
        st.divider()
        st.subheader("AI-Powered Analysis")
        
        analysis_col1, analysis_col2 = st.columns([1, 1])
        
        with analysis_col1:
            # Unified summary
            if st.button("Generate Unified Interpretation"):
                with st.spinner("Synthesizing commentaries..."):
                    commentaries = "\n\n".join(
                        f"{comm}: {verse['commentaries'][comm]}" 
                        for comm in selected_commentators 
                        if comm in verse['commentaries']
                    )
                    
                    prompt = f"""Synthesize a unified interpretation of Bhagavad Gita {verse_id} 
                    based on these commentaries:
                    
                    {commentaries}
                    
                    The seeker's original concern was: "{user_problem}"
                    
                    Provide a comprehensive understanding that integrates the different perspectives:"""
                    
                    response = asyncio.run(counselor.generate_ai_response(prompt))
                    st.markdown(response)
            
            # Modern analogy
            if st.button("Generate Modern Analogy"):
                with st.spinner("Creating contemporary explanation..."):
                    prompt = f"""Explain Bhagavad Gita {verse_id}:
                    {verse['text']}
                    
                    Using a modern analogy or real-world example that would help someone dealing with:
                    {user_problem}"""
                    
                    response = asyncio.run(counselor.generate_ai_response(prompt))
                    st.markdown(response)
        
        with analysis_col2:
            # Practical application
            if st.button("Suggest Practical Application"):
                with st.spinner("Creating actionable guidance..."):
                    prompt = f"""Based on Bhagavad Gita {verse_id}:
                    {verse['text']}
                    
                    Suggest 3-5 practical steps or exercises someone could take to apply this teaching
                    to their situation of: {user_problem}
                    
                    Format as a numbered list with concrete actions:"""
                    
                    response = asyncio.run(counselor.generate_ai_response(prompt))
                    st.markdown(response)
            
            # Related verses
            if st.button("Find Related Verses"):
                with st.spinner("Discovering thematic connections..."):
                    prompt = f"""Verse: {verse['text']}
                    
                    Identify 3-5 other verses from the Bhagavad Gita that address similar themes
                    or provide complementary teachings. For each, briefly explain the connection.
                    
                    Format as:
                    1. [verse reference] - [connection explanation]"""
                    
                    response = asyncio.run(counselor.generate_ai_response(prompt))
                    st.markdown(response)
        
        # Alternative Verses
        st.divider()
        st.subheader("Other Relevant Verses")
        
        if 'relevant_verses' in st.session_state:
            for i, verse_info in enumerate(st.session_state.relevant_verses[1:4]):
                verse_id = verse_info['id']
                verse_text = verse_info['text']
                
                with st.expander(f"Bhagavad Gita {verse_id}"):
                    st.markdown(f'<div class="verse-text">{verse_text}</div>', unsafe_allow_html=True)
                    
                    if st.button(f"Select this verse", key=f"alt_{i}"):
                        st.session_state.selected_verse = verse_id
                        st.rerun()

if __name__ == "__main__":
    main() 