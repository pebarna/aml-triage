import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = None

def _get_embedding_model() -> SentenceTransformer:
  global _EMBEDDING_MODEL
  if _EMBEDDING_MODEL is None:
    _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
  return _EMBEDDING_MODEL

def _load_corpus(corpus_path): 
  return json.loads(Path(corpus_path).read_text())

def _documents(corpus):
  return [f"{entry['title']}. {entry['text']}" for entry in corpus]

def _tfidf_scores(query, documents):
  vectorizer = TfidfVectorizer()
  doc_matrix = vectorizer.fit_transform(documents)
  query_vector = vectorizer.transform([query])
  return cosine_similarity(query_vector, doc_matrix)[0]

def _embedding_scores(query: str, documents: list[str]) -> np.ndarray: 
  model = _get_embedding_model()
  doc_embeddings = model.encode(documents)
  query_embedding = model.encode([query])
  return cosine_similarity(query_embedding, doc_embeddings)[0]

def top_k_typologies(query: str, k: int = 3, corpus_path: str | None = None) -> list[dict]:
  corpus = _load_corpus(corpus_path or "data/typologies.json")
  scores = _tfidf_scores(query, _documents(corpus))
  results = [
    {"id": entry["id"], "title": entry["title"], "text": entry["text"], "score": float(score)}
    for entry, score in zip(corpus, scores)
  ]
  results.sort(key=lambda r: r["score"], reverse=True)
  return results[:k]
  
def top_k_typologies_hybrid(query: str, k: int = 3, corpus_path: str | None = None, alpha: float = 0.5) -> list[dict]:
  corpus = _load_corpus(corpus_path or "data/typologies.json")
  documents = _documents(corpus)
  tfidf = _tfidf_scores(query, documents)
  embedding = _embedding_scores(query, documents)
  blended = alpha * tfidf + (1 - alpha) * embedding
  results = [
    {"id": entry["id"], "title": entry["title"], "text": entry["text"], "score": float(score)}
    for entry, score in zip(corpus, blended)
  ]
  results.sort(key=lambda r: r["score"], reverse=True)
  return results[:k]
  
  
