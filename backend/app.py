from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from xml.etree import ElementTree as ET

import ollama
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
SESSIONS_DIR = DATA_DIR / "sessions"

CHAT_MODEL = "llama3.2:1b"
EMBED_MODEL = "embeddinggemma:latest"
MAX_CHARS_PER_CHUNK = 900
CHUNK_OVERLAP = 120
MIN_CONTEXT_SCORE = 0.12

for folder in (DOCS_DIR, SESSIONS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Study Buddy Offline Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://0.0.0.0:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1)


class QuizGenerateRequest(BaseModel):
    session_id: str
    num_questions: int = Field(default=5, ge=1, le=10)


class QuizGradeRequest(BaseModel):
    session_id: str
    answers: list[Any]


def session_file(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def session_index_file(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}_index.json"


def load_session(session_id: str) -> dict[str, Any]:
    path = session_file(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(session_id: str, payload: dict[str, Any]) -> None:
    session_file(session_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_index(session_id: str, payload: dict[str, Any]) -> None:
    session_index_file(session_id).write_text(json.dumps(payload), encoding="utf-8")


def load_index(session_id: str) -> dict[str, Any]:
    path = session_index_file(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session index not found")
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_text(text: str, size: int = MAX_CHARS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".odt":
        try:
            with zipfile.ZipFile(path, "r") as archive:
                xml_data = archive.read("content.xml")
            root = ET.fromstring(xml_data)
            text_nodes = [node.text.strip() for node in root.iter() if node.text and node.text.strip()]
            return "\n".join(text_nodes)
        except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
            raise HTTPException(status_code=400, detail="Could not parse ODT file.") from exc

    raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, TXT, DOCX, or ODT.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    res = ollama.embed(model=EMBED_MODEL, input=texts)
    return res["embeddings"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def build_context(session_id: str, question: str, n_results: int = 5) -> list[dict[str, Any]]:
    query_embedding = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    index_data = load_index(session_id)
    source = str(index_data.get("source", "uploaded_document"))
    items = index_data.get("items", [])

    scored: list[dict[str, Any]] = []
    for item in items:
        emb = item.get("embedding")
        text = item.get("text")
        chunk_index = item.get("chunk_index")
        if isinstance(emb, list) and isinstance(text, str) and isinstance(chunk_index, int):
            score = cosine_similarity(query_embedding, emb)
            scored.append(
                {
                    "score": score,
                    "text": text,
                    "chunk_index": chunk_index,
                    "source": str(item.get("source", source)),
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_results]


def extract_json_array(raw_text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        return []
    return []


def safe_snippet(text: str, limit: int = 220) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def is_doc_name_question(text: str) -> bool:
    q = text.lower().strip()
    patterns = [
        r"\bname of (the )?document\b",
        r"\bwhat is the document name\b",
        r"\bfile name\b",
        r"\bwhat did i upload\b",
    ]
    return any(re.search(p, q) for p in patterns)


def is_doc_overview_question(text: str) -> bool:
    q = text.lower().strip()
    patterns = [
        r"\bwhat do you know\b",
        r"\bwhat is (this|the) document about\b",
        r"\bsummar(y|ize)\b",
        r"\boverview\b",
        r"\bmain idea\b",
    ]
    return any(re.search(p, q) for p in patterns)


def get_overview_context(session_id: str, max_items: int = 3) -> list[dict[str, Any]]:
    index_data = load_index(session_id)
    source = str(index_data.get("source", "uploaded_document"))
    items = index_data.get("items", [])
    contexts: list[dict[str, Any]] = []
    for item in items[:max_items]:
        text = item.get("text")
        chunk_index = item.get("chunk_index")
        if isinstance(text, str) and isinstance(chunk_index, int):
            contexts.append(
                {
                    "score": 1.0,
                    "text": text,
                    "chunk_index": chunk_index,
                    "source": str(item.get("source", source)),
                }
            )
    return contexts


def normalize_quiz_item(item: dict[str, Any]) -> dict[str, Any] | None:
    question = str(item.get("question", "")).strip()
    qtype = str(item.get("question_type", "mcq")).strip().lower()
    explanation = str(item.get("explanation", "")).strip()
    if not question:
        return None

    if qtype == "short_answer":
        answer_text = str(item.get("answer_text", "")).strip()
        if not answer_text:
            return None
        return {
            "question_type": "short_answer",
            "question": question,
            "answer_text": answer_text,
            "explanation": explanation or "Review the relevant section of the uploaded document.",
        }

    options = item.get("options", [])
    answer_index = item.get("answer_index", -1)
    if isinstance(options, list) and len(options) == 4 and answer_index in {0, 1, 2, 3}:
        return {
            "question_type": "mcq",
            "question": question,
            "options": [str(opt) for opt in options],
            "answer_index": int(answer_index),
            "explanation": explanation or "Answer supported by document context.",
        }
    return None


def fallback_quiz(context_text: str, num_questions: int) -> list[dict[str, Any]]:
    base_snippet = safe_snippet(context_text, 140) or "uploaded content"
    quiz: list[dict[str, Any]] = []
    for i in range(num_questions):
        if i % 2 == 0:
            quiz.append(
                {
                    "question_type": "mcq",
                    "question": f"Which statement best matches the document context ({i + 1})?",
                    "options": [
                        f"A key point is: {base_snippet}",
                        "The document is mostly about unrelated social media trends.",
                        "The document has no meaningful study material.",
                        "The document is only weather data.",
                    ],
                    "answer_index": 0,
                    "explanation": "Option 1 aligns with retrieved context snippets.",
                }
            )
        else:
            quiz.append(
                {
                    "question_type": "short_answer",
                    "question": f"In one sentence, summarize a main idea from the document ({i + 1}).",
                    "answer_text": base_snippet,
                    "explanation": "A correct answer should reference one of the document's main ideas.",
                }
            )
    return quiz


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), session_id: str | None = Form(default=None)
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name missing")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt", ".odt"}:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, DOCX, and ODT are supported")

    if session_id:
        session_payload = load_session(session_id)
        index_payload = load_index(session_id)
    else:
        session_id = uuid4().hex
        session_payload = {
            "session_id": session_id,
            "document_name": file.filename,
            "document_names": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
            "last_quiz": [],
        }
        index_payload = {"session_id": session_id, "source": "multiple_documents", "items": []}

    saved_name = f"{session_id}_{uuid4().hex[:8]}_{Path(file.filename).name}"
    save_path = DOCS_DIR / saved_name
    save_path.write_bytes(await file.read())

    text = extract_text(save_path)
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in document")

    embeddings = embed_texts(chunks)
    existing_items = index_payload.get("items", [])
    start_index = len(existing_items)
    new_items = [
        {
            "chunk_index": start_index + i,
            "text": chunks[i],
            "embedding": embeddings[i],
            "source": file.filename,
        }
        for i in range(len(chunks))
    ]
    index_payload["items"] = existing_items + new_items
    save_index(session_id, index_payload)

    document_names = session_payload.get("document_names")
    if not isinstance(document_names, list):
        existing_name = session_payload.get("document_name")
        document_names = [existing_name] if isinstance(existing_name, str) and existing_name else []
    if file.filename not in document_names:
        document_names.append(file.filename)
    session_payload["document_names"] = document_names
    session_payload["document_name"] = file.filename
    session_payload["stored_path"] = str(save_path)
    save_session(session_id, session_payload)

    return {
        "session_id": session_id,
        "uploaded_document": file.filename,
        "document_name": file.filename,
        "document_names": document_names,
        "documents_count": len(document_names),
        "chunks_indexed": len(chunks),
        "chunks_added": len(chunks),
        "total_chunks": len(index_payload["items"]),
        "message": "Document uploaded and indexed",
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    session = load_session(req.session_id)
    question_text = req.question.strip()

    if is_doc_name_question(question_text):
        answer = f'The uploaded document is "{session.get("document_name", "Unknown file")}".'
        session["messages"].append({"role": "user", "content": req.question})
        session["messages"].append({"role": "assistant", "content": answer})
        save_session(req.session_id, session)
        return {
            "answer": answer,
            "context_hits": 1,
            "sources": [
                {
                    "source": "session_metadata",
                    "chunk_index": -1,
                    "snippet": answer,
                    "score": 1.0,
                }
            ],
            "no_answer": False,
        }

    if is_doc_overview_question(question_text):
        contexts = get_overview_context(req.session_id, max_items=3)
    else:
        contexts = build_context(req.session_id, req.question, n_results=5)
    best_score = contexts[0]["score"] if contexts else -1.0
    context_chunks = [c["text"] for c in contexts]
    context_text = "\n\n".join(context_chunks)

    if not contexts or best_score < MIN_CONTEXT_SCORE:
        answer = "I could not find a clear answer in your uploaded document."
        session["messages"].append({"role": "user", "content": req.question})
        session["messages"].append({"role": "assistant", "content": answer})
        save_session(req.session_id, session)
        return {"answer": answer, "context_hits": 0, "sources": [], "no_answer": True}

    system_prompt = (
        "You are Study Buddy. Answer only with the provided context from the uploaded document. "
        "If the answer is not in context, say you do not find it in the document. "
        "Keep answers concise and study-friendly."
    )

    history = session.get("messages", [])[-8:]
    llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in history:
        llm_messages.append({"role": m["role"], "content": m["content"]})

    llm_messages.append(
        {
            "role": "user",
            "content": f"Document context:\n{context_text}\n\nQuestion: {req.question}",
        }
    )

    response = ollama.chat(model=CHAT_MODEL, messages=llm_messages)
    answer = response["message"]["content"].strip()

    session["messages"].append({"role": "user", "content": req.question})
    session["messages"].append({"role": "assistant", "content": answer})
    save_session(req.session_id, session)

    source_refs = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "snippet": safe_snippet(c["text"]),
            "score": round(float(c["score"]), 3),
        }
        for c in contexts[:3]
    ]
    return {"answer": answer, "context_hits": len(context_chunks), "sources": source_refs, "no_answer": False}


@app.post("/quiz/generate")
def generate_quiz(req: QuizGenerateRequest) -> dict[str, Any]:
    session = load_session(req.session_id)
    recent_chat = "\n".join(
        f"{m['role']}: {m['content']}" for m in session.get("messages", [])[-10:]
    )
    retrieval_context = build_context(req.session_id, recent_chat or "main topics", n_results=8)
    context_text = "\n\n".join(c["text"] for c in retrieval_context)

    prompt = (
        "Create a study quiz as strict JSON array only. "
        f"Return exactly {req.num_questions} objects. "
        "Mix question types: around half mcq and half short_answer. "
        "For mcq objects, use: question_type='mcq', question, options (exactly 4 strings), answer_index (0-3), explanation. "
        "For short answers, use: question_type='short_answer', question, answer_text, explanation. "
        "No markdown, no extra text."
        f"\n\nDocument context:\n{context_text}\n\nRecent chat:\n{recent_chat}"
    )

    raw = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])["message"][
        "content"
    ]

    parsed = extract_json_array(raw)
    quiz: list[dict[str, Any]] = []

    for item in parsed:
        normalized = normalize_quiz_item(item)
        if normalized is not None:
            quiz.append(normalized)
        if len(quiz) == req.num_questions:
            break

    if len(quiz) < req.num_questions:
        fallback = fallback_quiz(context_text, req.num_questions)
        needed = req.num_questions - len(quiz)
        quiz.extend(fallback[:needed])

    session["last_quiz"] = quiz
    save_session(req.session_id, session)

    public_quiz = []
    for i, q in enumerate(quiz):
        public_item: dict[str, Any] = {
            "question_number": i + 1,
            "question_type": q["question_type"],
            "question": q["question"],
        }
        if q["question_type"] == "mcq":
            public_item["options"] = q["options"]
        public_quiz.append(public_item)

    return {"quiz": public_quiz, "count": len(public_quiz)}


@app.post("/quiz/grade")
def grade_quiz(req: QuizGradeRequest) -> dict[str, Any]:
    session = load_session(req.session_id)
    quiz = session.get("last_quiz", [])

    if not quiz:
        raise HTTPException(status_code=400, detail="No generated quiz found. Run /quiz/generate first.")

    total = len(quiz)
    score = 0
    feedback: list[str] = []
    details: list[dict[str, Any]] = []

    for i, question in enumerate(quiz):
        qtype = question.get("question_type", "mcq")
        user_answer = req.answers[i] if i < len(req.answers) else -1

        if qtype == "short_answer":
            expected = str(question.get("answer_text", "")).strip().lower()
            user_text = str(user_answer).strip().lower() if isinstance(user_answer, str) else ""
            expected_words = [w for w in re.findall(r"[a-z0-9]+", expected) if len(w) > 3]
            match_count = sum(1 for w in set(expected_words[:10]) if w in user_text)
            is_correct = match_count >= 2 if expected_words else bool(user_text)
            if is_correct:
                score += 1
            explanation = question.get("explanation", "")
            feedback.append(f"Q{i + 1}: {'Correct' if is_correct else 'Needs improvement'}")
            details.append(
                {
                    "question_number": i + 1,
                    "question_type": "short_answer",
                    "correct": is_correct,
                    "expected": question.get("answer_text", ""),
                    "explanation": explanation,
                }
            )
        else:
            correct = int(question["answer_index"])
            selected = int(user_answer) if isinstance(user_answer, int) else -1
            is_correct = selected == correct
            if is_correct:
                score += 1
            explanation = question.get("explanation", "")
            feedback.append(
                f"Q{i + 1}: {'Correct' if is_correct else f'Incorrect. Correct option is {correct + 1}'}"
            )
            details.append(
                {
                    "question_number": i + 1,
                    "question_type": "mcq",
                    "correct": is_correct,
                    "correct_option": correct,
                    "explanation": explanation,
                }
            )

    return {
        "score": score,
        "total": total,
        "percent": round((score / total) * 100, 2) if total else 0,
        "feedback": feedback,
        "details": details,
    }
