from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import LLMChain
from streamlit_callback import StreamlitCallbackHandler


class Text_Expert:
    def __init__(self, inputs, prompt_from_template, temperture):
               
        self.system_prompt = self.get_system_prompt(inputs, prompt_from_template)

        self.user_prompt = HumanMessagePromptTemplate.from_template("{user_question}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )

        self.chat = ChatAnthropic(model='claude-v1-100k', temperature =temperture, max_tokens_to_sample=1024, streaming=True, callbacks=[StreamlitCallbackHandler()])

        self.chain = LLMChain(llm=self.chat, prompt=full_prompt_template)
        
        # st.write(full_prompt_template)
        
        # st.write("context01: ", st.session_state.context_01)  
        
        # st.write("context02: ", st.session_state.context_02)  
             
    def get_system_prompt(self, inputs, prompt_from_template):
        if self._default_prompt(prompt_from_template) != self._user_modified_prompt(inputs):
            system_prompt = self._user_modified_prompt(inputs)
        else:
            system_prompt = self._default_prompt(prompt_from_template)
        system_prompt = '"""' + system_prompt+ '"""'
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
    
    def run_qa_retrieval_chain(self, question, vectorstore):
        memory = ConversationBufferWindowMemory(k=5,memory_key='chat_history', return_messages=True, output_key='answer')
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 12})
        qa_retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat, retriever=retriever, memory=memory, return_source_documents=True
            )
        result = qa_retrieval_chain({"question":question})
        reference_docs = []
        for doc in result['source_documents'][0:5]:
            if doc.metadata['source'] not in reference_docs:
                reference_docs.append(doc.metadata['source'])
        return result['answer'], reference_docs
