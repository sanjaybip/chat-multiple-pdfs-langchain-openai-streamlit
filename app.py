from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.openai import OpenAI

from template import css, bot_template, user_template

import streamlit as st
import os

load_dotenv()

def get_pdf_text(pdf_dcos):
    raw_text = ""
    for pdf_doc in pdf_dcos:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    return raw_text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation

def get_answer(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="OpenAI query multiple PDFs at once", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs 📖")
    user_question = st.text_input(label="Ask question from uploaded PDFs")

    if user_question:
        get_answer(user_question)
    

    with st.sidebar:
        st.subheader("Your documents")
        pdf_dcos = st.file_uploader("Upload your PDF document", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("In process..."):
                #get texts from pdf
                extracted_text = get_pdf_text(pdf_dcos)
                
                #get chunks
                text_chunks = get_text_chunks(extracted_text)

                #get embedding and vectorstore
                vectorstore = get_vectorstore(text_chunks)

                #conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)



if __name__ == '__main__':
    main()