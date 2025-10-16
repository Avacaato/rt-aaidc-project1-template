# RAG-Based AI Assistant 

## 🤖 Project Overview

This repository provides a template for building a Retrieval-Augmented Generation (RAG) AI assistant that can answer questions using your own documents.
It combines search and NLP to generate meaningful answers leveraging the content you provide.

**Think of it as:** ChatGPT that knows about YOUR documents and can answer questions about them.


---
🚦 Implementation Plan

### Step 1: Prepare Your Documents

**Replace the sample documents with your own content**

The `data/` directory contains sample files on various topics. Replace these with documents relevant to your domain:

```
data/
├── your_topic_1.txt
├── your_topic_2.txt
└── your_topic_3.txt
```

Each file should contain text content you want your RAG system to search through.

---

## 🧪 Testing Your Implementation

### Test Individual Components

1. **Test chunking:**

   ```python
   from src.vectordb import VectorDB
   vdb = VectorDB()
   chunks = vdb.chunk_text("Your test text here...")
   print(f"Created {len(chunks)} chunks")
   ```
2. **Test document loading:**

   ```python
   documents = [{"content": "Test document", "metadata": {"title": "Test"}}]
   vdb.add_documents(documents)
   ```
3. **Test search:**

   ```python
   results = vdb.search("your test query")
   print(f"Found {len(results['documents'])} results")
   ```

### Test Full System

Once implemented, run:

```bash
python src/app.py
```
After Cloning try this example question to get an idea:

Input (query):
```
- "List three common tasks performed in NLP pipelines."
```

Expected Output:
```
Three common NLP pipeline tasks are tokenization, part-of-speech tagging, and named entity recognition.
```
---

## 🔧 Implementation Freedom

**Important:** This template uses specific packages (ChromaDB, LangChain, HuggingFace Transformers) and approaches, but **you are completely free to use whatever you prefer!**


---

## 🚀 Setup Instructions

### Prerequisites

Before starting, make sure you have:

- Python 3.8 or higher installed
- An API key from **one** of these providers:
  - [OpenAI](https://platform.openai.com/api-keys) (most popular)
  - [Groq](https://console.groq.com/keys) (free tier available)
  - [Google AI](https://aistudio.google.com/app/apikey) (competitive pricing)
  - [Perplexity AI](https://https://api.perplexity.ai) (free tier available)

### Quick Setup

1. **Clone and install dependencies:**

   ```bash
   git clone [(https://github.com/Avacaato/rt-aaidc-project1-template.git)]
   cd rt-aaidc-project1-template
   pip install -r requirements.txt
   ```

2. **Configure your API key:**

   ```bash
   # Create environment file (choose the method that works on your system)
   cp .env.example .env    # Linux/Mac
   copy .env.example .env  # Windows
   ```

   Edit `.env` and add your API key:

   ```
   OPENAI_API_KEY=your_key_here
   # OR
   GROQ_API_KEY=your_key_here  
   # OR
   GOOGLE_API_KEY=your_key_here
   # OR
   PERPLEXITY_API_KEY=your_key_here
   ```


---

## 📁 Project Structure

```
RT-AAIDC-PROJECT1-TEMPLATE/
├── data/                 # Place your documents here
├── src/                  # Application source code
│   └── app.py            # Main application logic
│   └── vectordb.py       # Vector database and search logic
├── .gitignore            # Git ignore rules
├── LICENSE               # Project license
├── README.md             # This documentation
├── requirements.txt      # Python dependenciesronment template
└── README.md             # This guide
```


---

## 🏁 Success Criteria

Your implementation is complete when:

1. ✅ You can load your own documents
2. ✅ The system chunks and embeds documents
3. ✅ Search returns relevant results
4. ✅ The RAG system generates contextual answers
5. ✅ You can ask questions and get meaningful responses

