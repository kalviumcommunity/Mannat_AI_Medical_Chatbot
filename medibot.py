import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

def choose_prompting_method(input_type, context, question, examples=None):
    """
    Selects prompting method and formats the prompt string.
    input_type: 'zero-shot', 'one-shot', 'few-shot', or 'dynamic'
    context: str, the context for the prompt
    question: str, the user's question
    examples: list of dicts, each with 'context' and 'question' and 'answer' (for few-shot/dynamic)
    Returns: (method, prompt_string)
    """
    input_type = input_type.lower()
    if input_type == 'zero-shot':
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return 'zero-shot', prompt
    elif input_type == 'one-shot' and examples and len(examples) >= 1:
        ex = examples[0]
        prompt = (
            f"Example:\nContext: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
            f"Now answer the following:\nContext: {context}\nQuestion: {question}\nAnswer:"
        )
        return 'one-shot', prompt
    elif input_type == 'few-shot' and examples and len(examples) >= 2:
        shots = ""
        for ex in examples:
            shots += f"Example:\nContext: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        prompt = (
            f"{shots}Now answer the following:\nContext: {context}\nQuestion: {question}\nAnswer:"
        )
        return 'few-shot', prompt
    elif input_type == 'dynamic' and examples:
        # Dynamic: choose number of examples based on question length or other heuristic
        num = min(3, len(examples))
        shots = ""
        for ex in examples[:num]:
            shots += f"Example:\nContext: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        prompt = (
            f"{shots}Now answer the following:\nContext: {context}\nQuestion: {question}\nAnswer:"
        )
        return 'dynamic', prompt
    else:
        # Default to zero-shot
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return 'zero-shot', prompt

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()