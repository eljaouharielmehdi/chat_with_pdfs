import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from htmlTemplates import css, bot_template, user_template

load_dotenv()

APP_TITLE = "Chat with multiple PDFs"

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's uploaded "
    "PDF documents. Only use the information in the provided context to answer. "
    "If the answer isn't contained in the context, say you don't know rather "
    "than guessing."
)


def get_pdf_text_by_file(pdf_docs):
    """Extract text from each PDF, keeping it grouped by source file."""
    docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        docs.append((pdf.name, text))
    return docs


def get_text_chunks(docs):
    """Split each document's text into chunks, tagging each with its source file."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks, metadatas = [], []
    for filename, text in docs:
        if not text.strip():
            continue
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            metadatas.append({"source": filename})
    return chunks, metadatas


def get_vectorstore(chunks, metadatas):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)


def answer_question(vectorstore, question, chat_history):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in relevant_docs
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, content in chat_history[-6:]:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    )

    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(messages)

    sources = sorted({doc.metadata.get("source", "unknown") for doc in relevant_docs})
    return response.content, sources


def init_session_state():
    defaults = {
        "vectorstore": None,
        "chat_history": [],
        "processed_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_chat():
    for msg in st.session_state.chat_history:
        template = user_template if msg["role"] == "user" else bot_template
        content = msg["content"]
        if msg.get("sources"):
            content += f"<br><small>Sources: {', '.join(msg['sources'])}</small>"
        st.write(template.replace("{{MSG}}", content), unsafe_allow_html=True)


def handle_userinput(user_question):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process at least one PDF before asking questions.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    history_so_far = [
        (m["role"], m["content"]) for m in st.session_state.chat_history[:-1]
    ]

    try:
        with st.spinner("Thinking..."):
            answer, sources = answer_question(
                st.session_state.vectorstore, user_question, history_so_far
            )
    except Exception as exc:
        st.session_state.chat_history.pop()
        st.error(f"Something went wrong while generating a response: {exc}")
        return

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def process_documents(pdf_docs):
    docs = get_pdf_text_by_file(pdf_docs)
    chunks, metadatas = get_text_chunks(docs)

    if not chunks:
        st.error(
            "No extractable text was found in the uploaded PDFs. "
            "They may be scanned images that need OCR first."
        )
        return

    st.session_state.vectorstore = get_vectorstore(chunks, metadatas)
    st.session_state.processed_files = [name for name, _ in docs]
    st.session_state.chat_history = []
    st.success(f"Processed {len(docs)} file(s) into {len(chunks)} chunks.")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    init_session_state()

    st.header(f"{APP_TITLE} :books:")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning(
            "No OPENAI_API_KEY found. Add it to a .env file or your environment "
            "before processing PDFs or asking questions."
        )

    if st.session_state.chat_history:
        render_chat()

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Process", disabled=not pdf_docs):
            with st.spinner("Processing documents..."):
                try:
                    process_documents(pdf_docs)
                except Exception as exc:
                    st.error(f"Failed to process documents: {exc}")

        if st.session_state.processed_files:
            st.markdown("**Loaded documents:**")
            for name in st.session_state.processed_files:
                st.markdown(f"- {name}")

        if st.session_state.chat_history:
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()

            transcript = "\n\n".join(
                f"{'You' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in st.session_state.chat_history
            )
            st.download_button(
                "Download chat history",
                data=transcript,
                file_name="chat_history.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
