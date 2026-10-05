# MultiPDF Chat App

> Originally based on the tutorial on [YouTube](https://youtu.be/dXxQ0LR-3Hg), since modernized and hardened for real use.

## Introduction
------------
The MultiPDF Chat App is a Python application that lets you chat with multiple PDF documents. Ask questions in natural language and get answers grounded in the content of the documents you uploaded, with the source file(s) cited for each answer.

## How It Works
------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

1. **PDF Loading** — the app reads each uploaded PDF and extracts its text, per page.
2. **Text Chunking** — each page's text is split into overlapping chunks, tagged with its source file name and page number.
3. **Embeddings** — an OpenAI embedding model turns each chunk into a vector, stored in a local FAISS index.
4. **Retrieval** — when you ask a question, the most relevant chunks are retrieved from the index, trimmed to a token budget so large PDFs can't overflow the model's context window.
5. **Response Generation** — the retrieved chunks (plus recent chat history) are streamed through a chat model, which answers using only that context and cites the file(s)/page(s) it used.

## Features

- Chat with one or more PDFs at once; new uploads merge into the existing session instead of replacing it.
- Answers stream in token-by-token, with source + page citations shown underneath.
- Pick the chat model (gpt-4o-mini / gpt-4o / gpt-3.5-turbo) and temperature from the sidebar.
- Graceful handling of missing API keys, empty uploads, oversized files, and scanned PDFs with no extractable text (instead of crashing).
- Clear chat and download chat history as a `.txt` transcript.
- List of currently loaded documents in the sidebar, with a one-click reset.

## Dependencies and Installation
----------------------------
1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Obtain an API key from OpenAI and add it to a `.env` file in the project directory:
   ```
   OPENAI_API_KEY=your_secret_api_key
   ```

## Usage
-----
1. Make sure dependencies are installed and `OPENAI_API_KEY` is set (via `.env` or your shell environment).
2. Run the app with Streamlit:
   ```
   streamlit run app.py
   ```
3. The app opens in your browser. Upload one or more PDFs in the sidebar and click **Process**.
4. Once processing finishes, ask questions about the documents in the main chat box.

If `OPENAI_API_KEY` isn't set, the app still loads and tells you what's missing instead of crashing.

## Running with Docker
-------------------
```
docker build -t chat-with-pdfs .
docker run -p 8501:8501 --env-file .env chat-with-pdfs
```
Then open http://localhost:8501.

## Development
-----------
Run the test suite (covers the pure text-chunking logic; no API key required):
```
pip install pytest
pytest
```

## Contributing
------------
Feel free to fork and adapt this app to your own needs.

## License
-------
The MultiPDF Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).
