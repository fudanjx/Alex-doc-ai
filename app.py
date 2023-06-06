from langchain.chat_models import ChatAnthropic
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain import LLMChain
from langchain import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from streamlit_callback import StreamlitCallbackHandler
import streamlit as st
import PyPDF2
import anthropic
import os
import pandas as pd

##################################################################################
class Text_Expert:
    def __init__(self, inputs, prompt_from_template, temperture):
               
        self.system_prompt = self.get_system_prompt(inputs, prompt_from_template)

        self.user_prompt = HumanMessagePromptTemplate.from_template("{user_question}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )

        self.chat = ChatAnthropic(model='claude-v1-100k', temperature =temperture, max_tokens_to_sample=1024, streaming=True, callbacks=[StreamlitCallbackHandler()])

        self.chain = LLMChain(llm=self.chat, prompt=full_prompt_template)
                
    def get_system_prompt(self, inputs, prompt_from_template):
        if self._default_prompt(prompt_from_template) != self._user_modified_prompt(inputs):
            system_prompt = self._user_modified_prompt(inputs)
        else:
            system_prompt = self._default_prompt(prompt_from_template)
        return SystemMessagePromptTemplate.from_template(system_prompt)
    
    def _user_modified_prompt(self, inputs):
        user_modified_prompt=inputs
        return user_modified_prompt
    
    def _default_prompt(self, prompt_from_template):
        default_prompt = prompt_from_template
          
        return default_prompt
        
    def run_chain(self, language, context_01, context_02, question):
        return self.chain.run(
            language=language, context_01 = context_01 , context_02 =context_02, user_question=question
        )
    
################################################################################
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
        st.session_state.context_01 = retrieve_multi_pdf_text(pdf_file_01) 

def cv_upload(upload_name):
    # create a upload file widget for a pdf
    pdf_file_02 = st.file_uploader(upload_name, type=["pdf"], accept_multiple_files=True)

   # if a pdf file is uploaded
    if pdf_file_02:
        st.session_state.context_02 = retrieve_multi_pdf_text(pdf_file_02)
        
def extract_info(data):
    result = ""
    for item in data:
        if item["type"] == "human":
            result += 'Human: '+ item["data"]["content"] + " ; \n\n"
        elif item["type"] == "ai":
            result += 'AI: '+ item["data"]["content"] + "\n\n"
    return result

    
####################################################################################        
st.set_page_config(page_title="Bot Alex!",page_icon="👀")    
# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/    
# create a streamlit app
st.title("🔎 Bot Alex - Doc AI")
with st.expander("###### Instructions"):
    st.write("Capable of analyzing any document.  You may refresh the page to start over")
    st.write("All the information in the conversation will be vanished after you close the session.")  
    st.write("You can download the chat history at the end of the conversation")

with st.expander("###### AI Model Setup"):
    anthropic.api_key = st.text_input("Enter Anthropic API Key", type="password")
    os.environ['ANTHROPIC_API_KEY']= anthropic.api_key   
    col1, col2 = st.columns([2,2])
    with col1:
        style = st.radio(
        "Style of the answer 👇",
        ('Deterministic', 'Balanced', 'Creative'))
        if style == 'Deterministic':
            temperature = 0.1
        if style == 'Balanced':
            temperature = 0.4
        if style == 'Creative':
            temperature = 0.8 
    
    with col2:  
        length = st.select_slider(
        'Length of the answer📏',
        options=['short', 'medium', 'long'])
        if length == 'short':
            max_token = "\n please try to answer within 200 words \n\n"
        if length == 'medium':
            max_token = "\n please try to answer within 500 words \n\n"
        if length == 'long':
            max_token = "\n please try to answer within 1000 words \n\n"

        lang = st.selectbox('Language Preference 🗣',
                            ('Professional', 'Legal', 'Simple', 'Chinese'))
        if lang == 'Professional':
            language = "\n please answer in Professional English \n\n"
        if lang == 'Legal':
            language = "\n please answer with Legal language \n\n"
        if lang == 'Simple':
            language = "\n please answer with simple English \n\n"
        if lang == 'Chinese':
            language = "\n please provide the answer in Chinese \n\n"
            
    if st.button("Enter"):
        if not anthropic.api_key:
            st.warning('Press enter after you iput the API key to apply', icon="⚠️")     
        st.session_state.temperature = temperature
        st.session_state.max_token = max_token
history = ChatMessageHistory()

####################################################################
#setup the sidebar section
with st.sidebar:
    # data_df_dase = pd.read_csv('prompt_template.csv')
    data_df_dase = pd.read_csv('https://www.dropbox.com/s/6v3ldwaoqe80iv0/prompt_template.csv?dl=1')

    prompt_category_list = data_df_dase['prompt_category'].tolist()

    option = st.selectbox(
        '#### Select Base Prompt:',
        prompt_category_list)

    df_selection = data_df_dase[data_df_dase['prompt_category'] == option]
    defult_prompt = df_selection['prompt'].values[0]
    fix_prompt = df_selection['fix_prompt'].values[0]
    upload_name1 = df_selection['Doc_01'].values[0]
    upload_name2 = df_selection['Doc_02'].values[0]
###################################################################
#setup the context input section
with st.expander("###### Upload your documents"):
    tab1, tab2 = st.tabs(["📂pdf doc  ", "📄  txt"])
    with tab1:      
        col1, col2 = st.columns([2,2])
        with col1:
            jd_upload(upload_name1)
        with col2:
            if type(upload_name2) != float:
                cv_upload(upload_name2)
            else:
                st.session_state.context_02 = "nothing here"
    with tab2:
        col1, col2 = st.columns([2,2])
        with col1:
            st.session_state.context_01 = st.text_area(upload_name1)
        with col2:
            if type(upload_name2) != float:
                st.session_state.context_02 = st.text_area(upload_name2)
            else:
                st.session_state.context_02 = "nothing here"
                
#######################################################################
#calling the langchain to run the model
if anthropic.api_key:

    if "Text_Expert" not in st.session_state:
        inputs =''
        st.session_state.Text_Expert = Text_Expert(inputs, defult_prompt, temperature)
        st.session_state.history = []      
 
    with st.sidebar:
        with st.expander("#### Modify Base Prompt"):
            inputs = st.text_area("modify_base_prompt",st.session_state.Text_Expert._default_prompt(prompt_from_template=defult_prompt), label_visibility="hidden")
        with st.expander("#### Review Base Prompt:"):
            user_final_prompt = inputs+fix_prompt+max_token+language
            user_final_prompt
            defult_prompt = defult_prompt + fix_prompt+max_token+language
            
    st.session_state.Text_Expert = Text_Expert(user_final_prompt,defult_prompt, temperature)

    
    with st.sidebar:
        if ("context_01" in st.session_state):
            # create a text input widget for a question
            question = st.text_area("##### Ask a question", label_visibility="visible")
            # create a button to run the model
            if st.button("Run"):
                # run the model
                bot_response = st.session_state.Text_Expert.run_chain(
                    'English', st.session_state.context_01, 
                        st.session_state.context_02, question)
                # st.session_state.bot_response = bot_response
                history.add_user_message(question)
                history.add_ai_message(bot_response)
                st.session_state.history +=history.messages
        dicts = messages_to_dict(st.session_state.history)
        string_hist = extract_info(dicts)
        if len(string_hist) != 0:
            st.download_button('Download Chat History', string_hist,'history.txt')
        else:
            pass

else:
    pass
