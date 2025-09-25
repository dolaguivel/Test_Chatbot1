# api_pdf_books.py
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import io
import re
import json
import uuid
import pdfplumber
from datetime import datetime
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Configuración base
# ------------------------
DATA_DIR = os.path.abspath("./data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
IDX_DIR = os.path.join(DATA_DIR, "index")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)

app = FastAPI(
    title="API Lector de Libros PDF",
    description="Sube, indexa y consulta libros en PDF (búsqueda, resumen y Q&A).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restringe dominios en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Utilidades
# ------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]", "_", name)
    return name[:150]

def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ------------------------
# Modelos de respuesta
# ------------------------
class DocMeta(BaseModel):
    id: str
    filename: str
    title: Optional[str] = None
    author: Optional[str] = None
    pages: int
    created_at: str

class SearchHit(BaseModel):
    page: int
    snippet: str
    score: float

class QAResponse(BaseModel):
    answer: str
    supporting_pages: List[int]

# ------------------------
# Índice en memoria
# ------------------------
class PDFIndex:
    """
    Administra PDFs, texto por página y vectores TF-IDF por documento.
    """
    def __init__(self):
        self.docs: Dict[str, Dict] = {}  # id -> {meta, pages_text, tfidf, vectorizer}
        self._bootstrap()

    def _bootstrap(self):
        # Carga metadatos existentes si están
        meta_db_path = os.path.join(IDX_DIR, "meta_db.json")
        meta_db = load_json(meta_db_path, {})
        # Reconstruye índices para PDFs presentes
        for doc_id, meta in meta_db.items():
            pdf_path = os.path.join(PDF_DIR, meta["filename"])
            if os.path.isfile(pdf_path):
                try:
                    self._ensure_index(doc_id, pdf_path, restore=True, meta=meta)
                except Exception:
                    # índice se reconstruye si falla
                    self.add_pdf_from_path(pdf_path, force_id=doc_id)

    def _store_meta(self, doc_id: str, meta: Dict):
        meta_db_path = os.path.join(IDX_DIR, "meta_db.json")
        meta_db = load_json(meta_db_path, {})
        meta_db[doc_id] = meta
        save_json(meta_db_path, meta_db)

    def add_pdf_from_upload(self, up: UploadFile) -> DocMeta:
        ext = (up.filename or "").lower().split(".")[-1]
        if ext != "pdf":
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")
        doc_id = str(uuid.uuid4())
        filename = f"{doc_id}__{safe_filename(up.filename)}"
        path = os.path.join(PDF_DIR, filename)
        with open(path, "wb") as f:
            f.write(up.file.read())
        return self._ensure_index(doc_id, path)

    def add_pdf_from_path(self, path: str, force_id: Optional[str] = None) -> DocMeta:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        doc_id = force_id or str(uuid.uuid4())
        filename = f"{doc_id}__{safe_filename(os.path.basename(path))}"
        dest = os.path.join(PDF_DIR, filename)
        if os.path.abspath(path) != os.path.abspath(dest):
            # Copia si viene de otra ruta
            with open(path, "rb") as src, open(dest, "wb") as dst:
                dst.write(src.read())
        return self._ensure_index(doc_id, dest)

    def _ensure_index(self, doc_id: str, pdf_path: str, restore: bool = False, meta: Optional[Dict] = None) -> DocMeta:
        # Extrae metadatos y texto por página
        pages_text: List[str] = []
        title = None
        author = None
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(normalize_text(text))
            # metadatos del PDF
            pdf_metadata = pdf.metadata or {}
            title = pdf_metadata.get("Title") or None
            author = pdf_metadata.get("Author") or None

        if meta is None:
            meta = {
                "id": doc_id,
                "filename": os.path.basename(pdf_path),
                "title": title or os.path.basename(pdf_path),
                "author": author,
                "pages": len(pages_text),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }

        # Construye TF-IDF por página para este documento
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="spanish", max_features=50000)
        tfidf = vectorizer.fit_transform(pages_text) if pages_text else None

        self.docs[doc_id] = {
            "meta": meta,
            "pages_text": pages_text,
            "vectorizer": vectorizer,
            "tfidf": tfidf,
        }
        self._store_meta(doc_id, meta)
        return DocMeta(**meta)

    def list_docs(self) -> List[DocMeta]:
        return [DocMeta(**d["meta"]) for d in self.docs.values()]

    def get_meta(self, doc_id: str) -> DocMeta:
        if doc_id not in self.docs:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        return DocMeta(**self.docs[doc_id]["meta"])

    def get_page(self, doc_id: str, page: int) -> str:
        if doc_id not in self.docs:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        pages = self.docs[doc_id]["pages_text"]
        if page < 1 or page > len(pages):
            raise HTTPException(status_code=400, detail=f"Página fuera de rango (1..{len(pages)})")
        return pages[page - 1]

    def get_pages_range(self, doc_id: str, start: int, end: int) -> List[str]:
        if doc_id not in self.docs:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        pages = self.docs[doc_id]["pages_text"]
        if start < 1 or end > len(pages) or start > end:
            raise HTTPException(status_code=400, detail=f"Rango inválido. Debe estar dentro de 1..{len(pages)} y start<=end")
        return pages[start - 1:end]

    def search(self, doc_id: str, query: str, k: int = 5) -> List[SearchHit]:
        if doc_id not in self.docs:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        entry = self.docs[doc_id]
        pages = entry["pages_text"]
        tfidf = entry["tfidf"]
        vec = entry["vectorizer"]

        if tfidf is None or vec is None or len(pages) == 0:
            return []

        q_vec = vec.transform([normalize_text(query)])
        sims = cosine_similarity(q_vec, tfidf).ravel()
        idxs = sims.argsort()[::-1][:k]
        hits: List[SearchHit] = []
        for i in idxs:
            score = float(sims[i])
            snippet = pages[i]
            # recorta snippet amigable
            snippet = (snippet[:320] + "…") if len(snippet) > 320 else snippet
            hits.append(SearchHit(page=i + 1, snippet=snippet, score=round(score, 4)))
        return hits

    def summarize_range(self, doc_id: str, start: int, end: int, max_sentences: int = 7) -> str:
        """
        Resumen extractivo muy simple: toma frases más representativas por TF-IDF.
        (Para producción, reemplazar por un modelo de resumen si lo deseas.)
        """
        pages = self.get_pages_range(doc_id, start, end)
        text = " ".join(pages)
        # segmenta en oraciones básicas por punto
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 0]
        if not sentences:
            return ""
        # TF-IDF por oración
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="spanish", max_features=20000)
        X = vec.fit_transform(sentences)
        scores = X.mean(axis=1).A.ravel()
        idx = scores.argsort()[::-1][:max_sentences]
        selected = [sentences[i] for i in sorted(idx)]  # manten orden aproximado
        return " ".join(selected)

    def qa(self, doc_id: str, question: str, k: int = 5) -> QAResponse:
        """
        Q&A básico tipo RAG: recupera k páginas más relevantes y arma una respuesta breve.
        """
        hits = self.search(doc_id, question, k=k)
        if not hits:
            return QAResponse(answer="No encontré contenido relevante para esa pregunta.", supporting_pages=[])
        # Heurística: concatena snippets y produce respuesta corta con las frases más similares
        context = " ".join([h.snippet for h in hits])
        # Extrae 2-3 oraciones que incluyan palabras de la pregunta
        q_terms = set([t for t in re.findall(r"\w{3,}", question.lower())])
        cand_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        scored = []
        for s in cand_sents:
            st = set(re.findall(r"\w{3,}", s.lower()))
            overlap = len(q_terms & st)
            scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:3]] or cand_sents[:2]
        answer = " ".join(top)
        supp = [h.page for h in hits[:min(len(hits), 5)]]
        return QAResponse(answer=answer or "No pude construir una respuesta útil.", supporting_pages=supp)

INDEX = PDFIndex()

# ------------------------
# Endpoints
# ------------------------

@app.api_route("/ping", methods=["GET", "HEAD"], summary="Verifica si la API está activa")
def ping():
    return {"status": "ok", "message": "API PDF operativa"}

@app.get("/pdfs", response_model=List[DocMeta], summary="Lista todos los PDFs")
def list_pdfs():
    return INDEX.list_docs()

@app.post("/pdfs/upload", response_model=DocMeta, summary="Sube e indexa un PDF")
async def upload_pdf(file: UploadFile = File(...)):
    meta = INDEX.add_pdf_from_upload(file)
    return meta

@app.get("/pdfs/{doc_id}/meta", response_model=DocMeta, summary="Metadatos del PDF")
def get_meta(doc_id: str):
    return INDEX.get_meta(doc_id)

@app.get("/pdfs/{doc_id}/pages/{page}", summary="Obtiene texto de una página (1-indexada)")
def get_page(doc_id: str, page: int):
    text = INDEX.get_page(doc_id, page)
    return {"doc_id": doc_id, "page": page, "text": text}

@app.get("/pdfs/{doc_id}/pages", summary="Obtiene texto de un rango de páginas")
def get_pages_range(doc_id: str,
                    start: int = Query(..., ge=1),
                    end: int = Query(..., ge=1)):
    pages = INDEX.get_pages_range(doc_id, start, end)
    return {"doc_id": doc_id, "start": start, "end": end, "pages": pages}

@app.get("/pdfs/{doc_id}/search", response_model=List[SearchHit], summary="Busca dentro del PDF")
def search_in_pdf(doc_id: str, q: str = Query(..., min_length=2), k: int = 5):
    return INDEX.search(doc_id, q, k=k)

class SummReq(BaseModel):
    start: int
    end: int
    max_sentences: int = 7

@app.post("/pdfs/{doc_id}/summarize", summary="Resumen extractivo simple del rango")
def summarize(doc_id: str, body: SummReq):
    summary = INDEX.summarize_range(doc_id, body.start, body.end, body.max_sentences)
    return {"doc_id": doc_id, "start": body.start, "end": body.end, "summary": summary}

class QAReq(BaseModel):
    question: str
    k: int = 5

@app.post("/pdfs/{doc_id}/qa", response_model=QAResponse, summary="Pregunta sobre el contenido del PDF (RAG simple)")
def qa(doc_id: str, body: QAReq):
    return INDEX.qa(doc_id, body.question, k=body.k)

# ------------------------
# Arranque local
# ------------------------
if __name__ == "__main__":
    uvicorn.run("api_pdf_books:app", host="0.0.0.0", port=8000, reload=True)

