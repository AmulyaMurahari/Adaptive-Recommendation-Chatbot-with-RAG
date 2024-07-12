import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_json_text(json_docs):
    text = ""
    for json_file in json_docs:
        data = json.load(json_file)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        text += f"{key}: {value}\n"
                else:
                    text += str(item) + "\n"
        elif isinstance(data, dict):
            for key, value in data.items():
                text += f"{key}: {value}\n"
        else:
            text += str(data) + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def embed_with_retries(embeddings, texts, max_retries=5):
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(texts)
        except google.generativeai.GenerativeAIError as e:
            if 'RATE_LIMIT_EXCEEDED' in str(e):
                st.warning(f"Rate limit exceeded, retrying in 10 seconds... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(10)
            else:
                raise
    raise RuntimeError("Failed to embed texts after multiple retries due to rate limits.")

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = []
    batch_size = 5  # Adjust batch size as needed
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        vectors.extend(embed_with_retries(embeddings, batch))
        time.sleep(2)  # Add a delay between batches to avoid rate limits
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="RAG")
    st.header("InsightBot AI 🤖")

    # File uploader
    pdf_docs = st.file_uploader("Upload your PDF or JSON Files", accept_multiple_files=True, type=["pdf", "json"])

    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                if pdf_docs[0].name.endswith(".pdf"):
                    raw_text = get_pdf_text(pdf_docs)
                elif pdf_docs[0].name.endswith(".json"):
                    raw_text = get_json_text(pdf_docs)
                else:
                    st.error("Unsupported file format. Please upload PDF or JSON files.")
                    return
                
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        else:
            st.warning("Please upload PDF or JSON files before processing.")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF or JSON Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
