# MultiPDF Chat App

> Originally based on the tutorial on [YouTube](https://youtu.be/dXxQ0LR-3Hg), since modernized and hardened for real use.

## Introduction
------------
The MultiPDF Chat App is a Python application that lets you chat with multiple PDF documents. Ask questions in natural language and get answers grounded in the content of the documents you uploaded, with the source file(s) cited for each answer.

## How It Works
------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

1. **PDF Loading** — the app reads each uploaded PDF and extracts its text, per file.
2. **Text Chunking** — extracted text is split into overlapping chunks, each tagged with its source file name.
3. **Embeddings** — an OpenAI embedding model turns each chunk into a vector, stored in a local FAISS index.
4. **Retrieval** — when you ask a question, the most relevant chunks are retrieved from the index.
5. **Response Generation** — the retrieved chunks (plus recent chat history) are passed to a chat model, which answers using only that context and cites which file(s) it used.

## Features

- Chat with one or more PDFs at once.
- Source citations shown under every answer.
- Graceful handling of missing API keys, empty uploads, and scanned PDFs with no extractable text (instead of crashing).
- Clear chat and download chat history as a `.txt` transcript.
- List of currently loaded documents in the sidebar.

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
