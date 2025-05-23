import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from src.helper import download_hugging_face_embeddings

from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(temperature=0.2, model_name="deepseek-r1-distill-llama-70b")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 Medical Chatbot")
st.write("Ask about diseases, causes and treatments..")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Your question:", key="input")

if st.button("Ask"):
    if user_input:
        st.session_state.chat_history.append(("You", user_input))
        response = rag_chain.invoke({"input": user_input})
        answer = response["answer"]
        st.session_state.chat_history.append(("Bot", answer))
        st.snow()

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**{sender}:** {message}")
    else:
        st.success(f"**{sender}:** {message}")

st.markdown(
    """
    <div style='position: fixed; bottom: 10px; right: 10px; background-color: white; 
                padding: 6px 12px; border-radius: 8px; box-shadow: 2px 2px 8px rgba(0,0,0,0.2); 
                font-size: 14px; font-family: sans-serif; color: black;'>
        Made with ❤️ by Baibhav Malviya
    </div>
    """,
    unsafe_allow_html=True
)
