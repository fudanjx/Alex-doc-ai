import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st

def retrieve_fin_hr_pcm_index():
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    folder_path="/Users/tammyauyong/Documents/Projects/OpenAI/Finance_HR_Procurement_faiss"
    #folder_path="https://www.dropbox.com/sh/o9tfhuad8uwqh4u/AAC-kEoP07FWzpI7PaVavcmka?dl=0"
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
