# Study Buddy Offline MVP - Build Record

## Project Goal
Build an offline-first Study Buddy app where a user uploads a document, chats about it, and takes a generated quiz.

## Core User Flow
1. User uploads a file (PDF/DOCX/TXT).
2. Backend extracts text and splits into chunks.
3. Chunks are embedded and stored in a local vector database.
4. User asks questions in a chat UI.
5. System retrieves relevant chunks and answers from local model context.
6. User clicks `Take Test`.
7. System generates quiz questions from document + chat context and grades answers.

## MVP Architecture
- Frontend: React app with Upload, Chat, and Quiz screens.
- Backend: FastAPI for upload/chat/quiz endpoints.
- LLM runtime: local Ollama for answer + quiz generation.
- Embeddings: local `gemma` embedding model via Ollama.
- Vector store: Chroma (persistent local storage).
- Local data folders:
  - `data/docs/`
  - `data/chroma/`
  - `data/sessions/`

## Will your local Ollama help?
Yes. It is required for the offline-first design.
- Ollama runs the local chat/generation model.
- No cloud API is needed for core functionality.

## Will gemma embeddings be used?
Yes.
- `gemma` embeddings convert document chunks and user queries into vectors.
- These vectors power semantic retrieval from Chroma.
- Retrieval is what makes chat answers and tests relevant to the uploaded document.

## Planned Backend Endpoints
- `POST /upload` -> parse + chunk + embed + store
- `POST /chat` -> retrieve + generate response
- `POST /quiz/generate` -> build quiz from doc/chat context
- `POST /quiz/grade` -> score and explain answers

## Next Build Steps
1. Scaffold `backend/` and `frontend/` directories.
2. Implement `/upload` with local parsing and embedding.
3. Implement `/chat` with retrieval + Ollama generation.
4. Build minimal chat UI.
5. Add `Take Test` generation + grading flow.

## UI MVP Status (Completed)
- Offline frontend MVP created in `frontend/` with:
  - document upload panel
  - chat interface
  - `Take Test` quiz flow
- Frontend is now wired to real backend endpoints at `http://localhost:8000`.

## Backend Status (Completed)
- `backend/app.py` created with:
  - `POST /upload`
  - `POST /chat`
  - `POST /quiz/generate`
  - `POST /quiz/grade`
  - `GET /health`
- Exact model names configured:
  - chat: `llama3.2:1b`
  - embeddings: `embeddinggemma:latest`
- Python dependencies listed in `backend/requirements.txt`.

## Install + Run
From project root:
```bash
cd /home/frenchie/Desktop/internet-money
source ain/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

## Run UI MVP
In another terminal:
```bash
cd /home/frenchie/Desktop/internet-money/frontend
python3 -m http.server 5500
```
Then open:
`http://localhost:5500`

## Note
- If `pip install` fails with DNS or network errors, retry once internet access is available.
