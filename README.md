# 🩺 Medical Chatbot  
A simple and effective medical chatbot powered by Hugging Face Transformers, LangChain, Faiss, and Streamlit. This project demonstrates how to build an AI-powered assistant that can answer health-related queries based on uploaded documents.

### 🚀 Features  
🔎 Document Search with Embeddings — Uses Hugging Face embeddings + Faiss for fast, relevant retrieval.

💬 Conversational Chatbot — Powered by a local LLM (e.g., Mistral), integrated via LangChain.

📄 PDF Support — Upload and embed your own medical documents or datasets.

🌐 Web Interface — Built with Streamlit for an interactive chat experience.

### 📁 Project Structure  
medical-chatbot/
│
├── data/                        # Folder to store your medical PDFs or documents
├── vectorstore/db_faiss/       # Stores generated vector embeddings
├── create_memory_for_llm.py    # Embeds data into the vector database
├── connect_memory_with_llm.py  # Connects vector store to the chatbot
├── medibot.py                  # Main Streamlit chatbot interface
├── medical-chatbot-ppt.pdf     # Presentation overview (architecture, flow)
├── Pipfile / Pipfile.lock      # Pipenv environment and dependencies
└── requirements.txt            # Python dependencies (alt to Pipfile)

### 🛠️ Installation  
✅ Python 3.8+ required  
✅ Recommended: Use a virtual environment (Pipenv or venv)  

Using Pipenv:  
```git clone https://github.com/kalviumcommunity/Mannat_AI_Medical_Chatbot.git```  
```cd Mannat_AI-Medical_Chatbot```  

#### Install dependencies  
pipenv install  

#### OR install specific packages manually  
```pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf huggingface_hub streamlit```  

#### How to Use  
Add Documents  
Place your medical PDFs inside the data/ folder.  

Generate Embeddings  
Run the script to create a vector store from the PDFs:  
```pipenv run python create_memory_for_llm.py```  

Run the Chatbot  
Launch the Streamlit interface:  
```pipenv run streamlit run medibot.py```  

### Use Cases  
Personal AI health assistant (non-clinical)  
FAQ bot for hospitals or clinics  
Medical student learning companion  