# CareerGuide AI — AI-Powered Career & Course Guidance Assistant

## 🌟 Project Overview
**CareerGuide AI** is an intelligent multimodal (text and voice) career counseling assistant built upon the **NCERT/CBSE Compendium of Academic Courses After 10+2**.

It features:
- **113+ Academic Streams**: Engineering, Medical & AYUSH, Science, Design, Arts, Law, Management, and more.
- **Smart Course Comparison**: Automatically detects comparison queries (e.g. *"Biotech vs Biomedical Engineering"*) and produces side-by-side analyses.
- **Transparent Source Citations**: Every response cites the course stream, category, and official page number in the NCERT handbook.
- **Multimodal Voice Interaction**: Real-time voice recording with Faster Whisper transcription and gTTS spoken audio feedback.
- **Vector Search & RAG**: HuggingFace (`all-MiniLM-L6-v2`) embeddings with Qdrant vector database and Groq Cloud Llama-3.1 inference.
- **RAGAS Evaluation Pipeline**: Offline semantic scoring measuring Faithfulness, Answer Correctness, Context Precision, and Context Recall.

---

## 🚀 Prerequisites

### 1. Environment Configuration (`.env`)
Make sure your `.env` file contains your Groq API key:
```ini
GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TEMPERATURE=0.0
GROQ_MAX_TOKENS=1024
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_URL=http://localhost:6333
```

> **Note**: If you don't run a Qdrant Docker container, the assistant automatically falls back to embedded local disk storage in `qdrant_storage/`.

---

## 🏃 Steps to Run

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Start the Web Application
```powershell
python app.py
```

### Step 3: Open the Web UI
Navigate to `http://localhost:5000` in your web browser.

---

## 💬 Sample Queries to Try

1. **Specific Course Inquiries:**
   - *"What are the eligibility criteria and career prospects in Artificial Intelligence & Machine Learning?"*
   - *"Tell me about Aeronautical Engineering and top institutes offering it."*
2. **Side-by-Side Comparisons:**
   - *"Compare Biotechnology Engineering vs Biomedical Engineering."*
   - *"What is the difference between Aeronautical and Aerospace Engineering?"*
3. **Interest-Based Counseling:**
   - *"I took Physics, Chemistry, and Biology in 12th. What career options do I have besides MBBS?"*
   - *"What subjects should I study in 10+2 to pursue a career in Design or Fine Arts?"*
4. **Voice Mode:**
   - Click the 🎤 microphone button in the Web UI, speak your question, click stop, and listen to the audio counselor response!

---

## 📊 Running Evaluation

To evaluate the assistant's retrieval and generation quality against benchmark datasets:
```powershell
python ragas_evaluation.py
```
This computes Faithfulness, Answer Correctness, Context Precision, and Context Recall, outputting a detailed summary to `ragas_results.json`.
