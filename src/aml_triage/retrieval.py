import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _load_corpus(corpus_path): 
  return json.loads(Path(corpus_path).read_text())

def _documents(corpus):
  return [f"{entry['title']}. {entry['text']}" for entry in corpus]

def _tfidf_scores(query, documents):
  vectorizer = TfidfVectorizer()
  doc_matrix = vectorizer.fit_transform(documents)
  query_vector = vectorizer.transform([query])
  return cosine_similarity(query_vector, doc_matrix)[0]

def top_k_typologies(query: str, k: int = 3, corpus_path: str | None = None) -> list[dict]:
  corpus = _load_corpus(corpus_path or "data/typologies.json")
  scores = _tfidf_scores(query, _documents(corpus))
  results = [
    {"id": entry["id"], "title": entry["title"], "text": entry["text"], "score": float(score)}
    for entry, score in zip(corpus, scores)
  ]
  results.sort(key=lambda r: r["score"], reverse=True)
  return results[:k]
  
  
  
