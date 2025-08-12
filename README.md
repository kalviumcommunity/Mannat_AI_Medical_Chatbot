# 🧠 Medical Chatbot  
A smart AI-powered Medical Chatbot that can read, understand, and respond to user queries using uploaded medical documents. Built using LLMs, vector databases, and streaming interfaces, this project simulates how AI can support basic healthcare communication by acting as a domain-aware assistant.

### 📌 Project Idea  
The goal is to build a domain-specific chatbot capable of:  
Reading medical PDFs (like clinical guidelines or disease documentation)  
Storing and understanding context using text embeddings  
Answering user questions based on the uploaded content  
Providing a simple, secure, and interactive interface  

This can be applied in:  
Hospitals (for answering repetitive queries)  
Medical education (student Q&A)  
Patient self-help tools  
This is an educational tool and not a replacement for licensed medical professionals.  

### 🔧 Technical Implementation  
Architecture Overview  

User ⟶ Streamlit Chat UI
          ⬇
     LangChain + LLM (Mistral)
          ⬇
     Faiss Vector Store
          ⬇
   Embedded PDF Documents (via HuggingFace)

### 🛠️ Tech Stack  
| Component          | Tool/Library             | Purpose                                 |
| ------------------ | ------------------------ | --------------------------------------- |
| Language Model     | Mistral (via LangChain)  | Generates intelligent responses         |
| Embedding Model    | HuggingFace Transformers | Converts text to numeric vectors        |
| Vector Store       | Faiss                    | Enables fast similarity search          |
| File Parsing       | PyPDF                    | Extracts raw text from uploaded PDFs    |
| Frontend Interface | Streamlit                | Provides user-friendly web interface    |
| Orchestration      | LangChain                | Chains memory, embedding, and LLM logic |


### 📂 Project Structure  

medical-chatbot/
├── data/                         # PDF documents go here
├── vectorstore/db_faiss/        # Preprocessed and indexed embeddings
├── medibot.py                   # Streamlit app to run the chatbot
├── create_memory_for_llm.py     # Script to build the vector index
├── connect_memory_with_llm.py   # Binds memory with chatbot
├── Pipfile / requirements.txt   # Project dependencies
└── medical-chatbot-ppt.pdf      # Project explanation slides


### ⚡ Efficiency  
Embedding and search use optimized CPU-based Faiss.  
Only top relevant documents are passed to the LLM, reducing inference cost.  
Runs locally with minimal hardware (can be adapted for GPUs or APIs).  

### 🌐 Scalability  

The modular architecture supports:  
Adding more documents or domains  
Switching to cloud-based LLMs or Faiss on GPU  
Expanding the frontend with file uploads, login, etc.  
Vector search scales well for thousands of documents without major performance loss.  

### How to Run  
Install dependencies:  
```pipenv install```

Add documents to /data.  

Generate memory:  
```pipenv run python create_memory_for_llm.py```

Start the chatbot:  
```pipenv run streamlit run medibot.py```

### License
This project is open for learning purposes. Do not use for actual medical diagnosis or emergency services.  