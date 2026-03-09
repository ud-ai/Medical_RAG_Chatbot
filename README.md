# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for medical Q&A using:
- MedQuAD dataset
- FAISS vector search
- Sentence Transformers embeddings
- Gemini models via LangChain
- Gradio web UI

## Features

- Medical Q&A grounded in MedQuAD context
- FAISS retrieval over local index files
- Automatic model fallback on Gemini rate-limit errors
- Manual model switch from UI dropdown
- Response caching to reduce repeated API calls
- Deployment-ready Gradio app (`app.py`)

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- .env                      # local only (ignored by git)
|-- medquad_2000.csv
|-- faiss_index/
|   |-- index.faiss
|   `-- index.pkl
|-- deployment_package/
|   |-- medquad_2000.csv
|   `-- faiss_index/
`-- Medical_RAG_Chatbot (1).ipynb
```

## Prerequisites

- Python 3.10+
- A valid Gemini API key

## Local Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create `.env` in the project root.

```env
GEMINI_API_KEY=your_api_key_here
# Optional
# GOOGLE_API_KEY=your_api_key_here
# PORT=7865
# FAISS_INDEX_DIR=./faiss_index
# DATASET_PATH=./medquad_2000.csv
```

4. Ensure data/index files exist:
- `./faiss_index/index.faiss`
- `./faiss_index/index.pkl`
- `./medquad_2000.csv`

If missing, copy from `deployment_package/`.

```powershell
Copy-Item -Path ".\deployment_package\faiss_index" -Destination ".\faiss_index" -Recurse -Force
Copy-Item -Path ".\deployment_package\medquad_2000.csv" -Destination ".\medquad_2000.csv" -Force
```

5. Run the app.

```powershell
python app.py
```

Then open `http://localhost:7865` (or the `PORT` you set).

## Deployment Notes

`app.py` is configured for cloud deployment:
- `server_name="0.0.0.0"`
- `server_port=int(os.environ.get("PORT", 7865))`
- `share=False`

### Render / Railway

- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
python app.py
```

- Required environment variable:
  - `GEMINI_API_KEY`

- Include these files in deployment:
  - `app.py`
  - `requirements.txt`
  - `medquad_2000.csv`
  - `faiss_index/` directory

### Hugging Face Spaces (Gradio)

- SDK: Gradio
- App file: `app.py`
- Add `GEMINI_API_KEY` in Space secrets
- Upload `faiss_index/` and `medquad_2000.csv`

## Security

- `.env` is ignored via `.gitignore`
- Do not commit API keys

## Troubleshooting

- Port in use: set `PORT` in `.env` (for example `PORT=7866`) and restart.
- Rate limit/quota exceeded: use model switch dropdown or wait for quota reset.
- Missing index/data files: copy from `deployment_package/`.
