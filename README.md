
# 🕉 Bhagavad Gita AI Counselor

![Bhagavad Gita AI Counselor]((https://github.com/akarshcodes/AI-Counselor/blob/main/Screenshot%20(274).png))

**Bhagavad Gita AI Counselor** is an intelligent spiritual guide powered by semantic search and AI reasoning. It lets users ask life questions and receive contextual guidance from verses, commentaries, and translations of the Bhagavad Gita, along with AI-generated explanations, analogies, and actionable steps.

---

##  Features

-  Semantic search for life questions using `ChromaDB` + `SentenceTransformer`
- Real-time access to Gita verses, commentaries, and translations
-  AI-powered explanations using `Ollama` with LLM models
-  Interactive concept maps using `Plotly` and `NetworkX`
- 🗣 Multi-commentator debate simulator
- Personalized modern analogies and practical applications
-  Thematic verse recommendations

---

##  Screenshots

| Home Page | Verse Display with Concept Map |
|----------|-------------------------------|
| ![home]((https://github.com/akarshcodes/AI-Counselor/blob/main/Screenshot%20(275).png)) | ![verse_view]((https://github.com/akarshcodes/AI-Counselor/blob/main/Screenshot%20(276).png)) |

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/gita-ai-counselor.git
   cd gita-ai-counselor
   ```

2. **Install dependencies:**

   Use a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

   Then install:

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

4. **Ensure Ollama is running locally:**

   This app uses `Ollama` at `http://localhost:11434`. Start it before launching the app:

   ```bash
   ollama run mistral
   ```

---

##  Data Folder Structure

Place your Bhagavad Gita JSON files inside a directory named `gita_dir` at the project root. Each JSON file should follow this format:

```json
{
  "BhagavadGitaChapter": [
    {
      "chapter": 1,
      "verse": 1,
      "text": "Dharma-kshetre kuru-kshetre...",
      "commentaries": {
        "Swami Ramsukhdas": "...",
        "Sri Ramanuja": "..."
      },
      "translations": {
        "swami sivananda": "..."
      }
    }
  ]
}
```

---

##  How It Works

- Uses **SentenceTransformer** to embed user queries, verses, and commentaries.
- Embeddings are stored in **ChromaDB** for efficient similarity search.
- Uses **async I/O** and **threading** to load and process large verse sets fast.
- When a verse is selected, AI can:
  - Debate as a commentator
  - Generate analogies
  - Suggest practical steps
  - Find related verses

---

##  Example `.env` (if you add it)

```dotenv
CHROMA_DB_PATH=./chroma_data
OLLAMA_ENDPOINT=http://localhost:11434/api/generate
```

---

##  Roadmap

- [ ] Add verse audio and Sanskrit chanting
- [ ] Support multiple languages
- [ ] Fine-tune QA generation on Gita data
- [ ] Host public demo with persistent DB

---

##  Contributing

1. Fork the repo and clone it
2. Create your feature branch: `git checkout -b my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin my-feature`
5. Open a pull request!

---

##  License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

##  Acknowledgements

- Bhagavad Gita open datasets
- Swamis and scholars for timeless wisdom
- Open-source AI community

---

##  Feedback and Contact

For feedback, suggestions, or collaborations, feel free to open an issue or reach out via [your-email@example.com].

---

##  Repository Tree(Preferrable since this beta and casual AI practice we ignore the struct.)

```bash
├── app.py
├── gita_dir/
│   ├── chapter1.json
├── images/
│   ├── home.png
│   ├── verse_view.png
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Link 

> Prefer this link locally for viewing: http://172.17.34.11:8502/
