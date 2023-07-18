import PyPDF2
import streamlit as st


####################################################################################  
def retrieve_multi_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()  
    return text

def jd_upload(upload_name):
    # create a upload file widget for a pdf
    pdf_file_01 = st.file_uploader(upload_name, type=["pdf"], accept_multiple_files=True)

    # if a pdf file is uploaded
    if pdf_file_01:
        return retrieve_multi_pdf_text(pdf_file_01) 
    else:
        return "" 

def cv_upload(upload_name):
    # create a upload file widget for a pdf
    pdf_file_02 = st.file_uploader(upload_name, type=["pdf"], accept_multiple_files=True)

   # if a pdf file is uploaded
    if pdf_file_02:
        return retrieve_multi_pdf_text(pdf_file_02)
    else:
        return ""  
#extract the message history dict into a string         
def extract_info(data):
    result = ""
    for item in data:
        if item["type"] == "human":
            result += 'Human: '+ item["data"]["content"] + " ; \n\n"
        elif item["type"] == "ai":
            result += 'AI: '+ item["data"]["content"] + "\n\n"
    return result

#extract the message dict of human input portion history into a list        
def extract_human_history(data):
    human_data = []
    for d in data:
        if d['type'] == 'human':
            human_data.append(d['data']['content'])
    return human_data

# Define a function that takes a list as input and returns the reversed list
def reverse_list(lst):
    return lst[::-1]

# Convert the list to a string, and add line break after each list item
def list_to_string(lst):
    return '\n\n'.join(str(i+1) + '. ' + str(item) for i, item in enumerate(lst))



####################################################################################  