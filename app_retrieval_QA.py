import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import streamlit as st
import common_functions as cf

def retrieve_fin_hr_pcm_index():
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    folder_path="Finance_HR_Procurement_faiss"
    #download_path="https://www.dropbox.com/sh/o9tfhuad8uwqh4u/AAC-kEoP07FWzpI7PaVavcmka?dl=0"
    vectorstore = FAISS.load_local(
        folder_path=folder_path, embeddings=embedding,
        index_name="fin_hr_procurement_HuggingFace"
        )
    return embedding, vectorstore


def display_reference(reference_docs):
    reference_str=""
    with st.expander("#### Reference Documents"):
        for doc_count in range(len(reference_docs)):
            doc = reference_docs[doc_count]
            reference_str += f"\n\nReference Doc {doc_count+1}: {doc}"
        st.write(reference_str)
    return reference_str

def retrieve_conversation_memory():
    memory = ConversationBufferMemory(
        memory_key='chat_history', input_key="question", return_messages=True, output_key='answer'
        )
    return memory

def upload_documents():
    # Upload Patient Notes
    context=""
    uploaded_files = st.file_uploader("Upload relevant documents for context", 
                                      type=["pdf", "xlsx", 'csv', 'xls', 'docx'], accept_multiple_files=True)
    if uploaded_files:
        count=0
        for uploaded_file in uploaded_files:
            count+=1
            if uploaded_file.name.endswith(('.pdf')):
                context += cf.retrieve_multi_pdf_text(upload_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls', '.csv')):
                extracted_text = cf.retrieve_multi_excel_text(upload_file)
                context += f"Document {count}:{extracted_text}"
    return context
