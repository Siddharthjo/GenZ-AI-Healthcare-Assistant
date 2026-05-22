"""
Semantic symptom matcher with two backends:

1. SentenceTransformer (preferred) — full semantic search with `all-MiniLM-L6-v2`
2. TF-IDF fallback               — character n-grams + word n-grams for typo tolerance

Both backends pre-compute embeddings / vectors at init time so per-request
inference is just a cosine similarity lookup.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSymptomMatcher:
    """Match free-text user input to the closest canonical symptom."""

    def __init__(self, symptom_list: list[str]):
        self.symptoms = symptom_list
        self._display = [s.replace('_', ' ') for s in symptom_list]
        self._backend = self._init_transformer() or self._init_tfidf()

    # ── Backend initialisation ────────────────────────────────────────────────

    def _init_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(self._display, show_progress_bar=False,
                                      convert_to_numpy=True)
            # L2-normalise so cosine sim = dot product
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return {'type': 'transformer', 'model': model, 'embeddings': embeddings}
        except Exception:
            return None

    def _init_tfidf(self):
        # Word n-grams (1-2) + character n-grams (2-4) for typo tolerance
        word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                                   token_pattern=r'[a-z]+')
        char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        word_mtx = word_vec.fit_transform(self._display)
        char_mtx = char_vec.fit_transform(self._display)
        return {
            'type': 'tfidf',
            'word_vec': word_vec, 'word_mtx': word_mtx,
            'char_vec': char_vec, 'char_mtx': char_mtx,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return self._backend['type']

    def match(self, user_input: str) -> str:
        """Return the canonical symptom name closest to `user_input`."""
        query = user_input.lower().replace('_', ' ').strip()
        if self._backend['type'] == 'transformer':
            return self._match_transformer(query)
        return self._match_tfidf(query)

    def match_many(self, user_input: str, top_k: int = 3) -> list[str]:
        """Return the top-k canonical symptoms closest to `user_input`."""
        query = user_input.lower().replace('_', ' ').strip()
        scores = self._score(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.symptoms[i] for i in top_indices]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _match_transformer(self, query: str) -> str:
        emb = self._backend['model'].encode([query], convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        scores = (self._backend['embeddings'] @ emb.T).flatten()
        return self.symptoms[int(np.argmax(scores))]

    def _match_tfidf(self, query: str) -> str:
        scores = self._score(query)
        return self.symptoms[int(np.argmax(scores))]

    def _score(self, query: str) -> np.ndarray:
        b = self._backend
        w_score = cosine_similarity(
            b['word_vec'].transform([query]), b['word_mtx']
        ).flatten()
        c_score = cosine_similarity(
            b['char_vec'].transform([query]), b['char_mtx']
        ).flatten()
        return 0.5 * w_score + 0.5 * c_score
