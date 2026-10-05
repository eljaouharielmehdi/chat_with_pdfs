import os

import streamlit as st
import tiktoken
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from htmlTemplates import css, bot_template, user_template

load_dotenv()

APP_TITLE = "Chat with multiple PDFs"
MAX_FILE_SIZE_MB = 20
MAX_CONTEXT_TOKENS = 6000

_token_encoding = None
_token_encoding_unavailable = False

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's uploaded "
    "PDF documents. Only use the information in the provided context to answer. "
    "If the answer isn't contained in the context, say you don't know rather "
    "than guessing."
)


def get_pdf_pages_by_file(pdf_docs):
    """Extract text per page, keeping each page's source file name and page number."""
    pages = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            extracted = page.extract_text()
            if extracted and extracted.strip():
                pages.append((pdf.name, page_number, extracted))
    return pages


def get_text_chunks(pages):
    """Split each page's text into chunks, tagging each with its source file and page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks, metadatas = [], []
    for filename, page_number, text in pages:
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            metadatas.append({"source": filename, "page": page_number})
    return chunks, metadatas


def format_source(metadata):
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    return f"{source} (p. {page})" if page else source


def get_vectorstore(chunks, metadatas):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)


MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


def count_tokens(text):
    """Count tokens with tiktoken, falling back to a rough estimate if its
    encoding data can't be loaded (e.g. no network access)."""
    global _token_encoding, _token_encoding_unavailable

    if not _token_encoding_unavailable and _token_encoding is None:
        try:
            _token_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _token_encoding_unavailable = True

    if _token_encoding is not None:
        return len(_token_encoding.encode(text))
    return max(1, len(text) // 4)


def select_context_docs(docs, max_tokens=MAX_CONTEXT_TOKENS):
    """Keep retrieved chunks within a token budget so large PDFs don't blow the model's context window."""
    selected, total = [], 0
    for doc in docs:
        doc_tokens = count_tokens(doc.page_content)
        if selected and total + doc_tokens > max_tokens:
            break
        selected.append(doc)
        total += doc_tokens
    return selected


def build_messages(vectorstore, question, chat_history):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = select_context_docs(retriever.invoke(question))

    context = "\n\n".join(
        f"[Source: {format_source(doc.metadata)}]\n{doc.page_content}"
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

    sources = sorted({format_source(doc.metadata) for doc in relevant_docs})
    return messages, sources


def stream_answer(vectorstore, question, chat_history, model, temperature):
    """Returns a (token generator, sources) pair for the given question."""
    messages, sources = build_messages(vectorstore, question, chat_history)
    llm = ChatOpenAI(model=model, temperature=temperature)

    def tokens():
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    return tokens(), sources


def init_session_state():
    defaults = {
        "vectorstore": None,
        "chat_history": [],
        "processed_files": [],
        "model": MODEL_OPTIONS[0],
        "temperature": 0.0,
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

    placeholder = st.empty()
    accumulated = ""
    try:
        token_stream, sources = stream_answer(
            st.session_state.vectorstore,
            user_question,
            history_so_far,
            st.session_state.model,
            st.session_state.temperature,
        )
        for token in token_stream:
            accumulated += token
            placeholder.write(
                bot_template.replace("{{MSG}}", accumulated + "▌"),
                unsafe_allow_html=True,
            )
    except Exception as exc:
        placeholder.empty()
        st.session_state.chat_history.pop()
        st.error(f"Something went wrong while generating a response: {exc}")
        return

    final_content = accumulated
    if sources:
        final_content += f"<br><small>Sources: {', '.join(sources)}</small>"
    placeholder.write(bot_template.replace("{{MSG}}", final_content), unsafe_allow_html=True)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": accumulated, "sources": sources}
    )


def process_documents(pdf_docs):
    already_loaded = set(st.session_state.processed_files)
    new_files = [pdf for pdf in pdf_docs if pdf.name not in already_loaded]

    if not new_files:
        st.info("All selected files are already loaded.")
        return

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    oversized = [pdf.name for pdf in new_files if pdf.size > max_bytes]
    if oversized:
        st.error(
            f"Skipping file(s) larger than {MAX_FILE_SIZE_MB}MB: " + ", ".join(oversized)
        )
        new_files = [pdf for pdf in new_files if pdf.name not in oversized]

    if not new_files:
        return

    pages = get_pdf_pages_by_file(new_files)
    chunks, metadatas = get_text_chunks(pages)

    extracted_names = {filename for filename, _, _ in pages}
    empty_files = [pdf.name for pdf in new_files if pdf.name not in extracted_names]
    if empty_files:
        st.warning(
            "No extractable text found in: " + ", ".join(empty_files) + ". "
            "They may be scanned images that need OCR first."
        )

    if not chunks:
        return

    new_vectorstore = get_vectorstore(chunks, metadatas)
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = new_vectorstore
    else:
        st.session_state.vectorstore.merge_from(new_vectorstore)

    loaded_names = [pdf.name for pdf in new_files if pdf.name in extracted_names]
    st.session_state.processed_files.extend(loaded_names)
    st.success(f"Processed {len(loaded_names)} new file(s) into {len(chunks)} chunks.")


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

            if st.button("Remove all documents"):
                st.session_state.vectorstore = None
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                st.rerun()

        st.subheader("Model settings")
        st.session_state.model = st.selectbox(
            "Chat model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.model)
        )
        st.session_state.temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1
        )

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
