"""Hash-based TF-IDF embedding module.

Produces dense, fixed-dimension embedding vectors from raw text using a
hash-trick approach that requires only ``numpy`` — no external API calls,
no heavy ML model downloads, and no internet access at runtime.

Algorithm overview
------------------
1. Tokenise the input text (lowercase, split on non-alphabetic characters).
2. Compute a term-frequency (TF) map for the token stream.
3. Project each token into a fixed-dimension vector using a deterministic
   hash function: bucket = hash(token) % dimension.  The TF weight is
   accumulated at the corresponding bucket index.
4. L2-normalise the resulting vector so that cosine similarity reduces to a
   simple dot product.

This is intentionally lightweight: the embedding is not semantically rich,
but it is sufficient for nearest-neighbour retrieval over a small domain-
specific knowledge base of fraud patterns where lexical overlap is a strong
retrieval signal.
"""

from __future__ import annotations

import re
from typing import ClassVar

import numpy as np


class SimpleEmbedding:
    """Lightweight hash-based TF-IDF embedding without external dependencies.

    Attributes:
        dimension: Fixed output dimensionality of all embedding vectors.

    Example::

        emb = SimpleEmbedding(dimension=128)
        vec = emb.embed("card skimming ATM withdrawal suspicious")
        assert vec.shape == (128,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-6
    """

    # Compile once at class level for efficiency.
    _TOKEN_RE: ClassVar[re.Pattern[str]] = re.compile(r"[^a-z]+")

    def __init__(self, dimension: int = 128) -> None:
        """Initialise the embedding.

        Args:
            dimension: Size of the output vector.  Higher values reduce hash
                collisions at the cost of memory.  Must be a positive integer.

        Raises:
            ValueError: If *dimension* is not a positive integer.
        """
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        self.dimension = dimension
        # Optional vocabulary built from a corpus (not required for hashing,
        # but populated by _build_vocab for callers that need it).
        self._vocab: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    def _build_vocab(self, texts: list[str]) -> None:
        """Build a term → index vocabulary from a corpus of texts.

        The vocabulary is *not* required for :meth:`embed` (which uses
        hashing), but it is made available for callers that wish to inspect
        or serialise the token space.

        Args:
            texts: Corpus of raw text strings.
        """
        vocab: dict[str, int] = {}
        for text in texts:
            for token in self._tokenize(text):
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocab = vocab

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalised dense vector.

        The returned vector has unit L2 norm (or is the zero vector if the
        input produces no tokens, which is extremely unlikely in practice).

        Args:
            text: Raw input text.  May contain punctuation and mixed case.

        Returns:
            Float64 numpy array of shape ``(self.dimension,)``.
        """
        tokens = self._tokenize(text)
        vector = np.zeros(self.dimension, dtype=np.float64)

        if not tokens:
            return vector

        # Term-frequency accumulation via hash bucketing.
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        total_tokens = len(tokens)
        for token, count in tf.items():
            bucket = self._hash_token(token)
            # Normalise TF by document length to reduce bias toward long docs.
            vector[bucket] += count / total_tokens

        return self._l2_normalize(vector)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into a 2-D matrix.

        Args:
            texts: List of raw input strings.  Empty list returns a
                zero-row matrix of shape ``(0, self.dimension)``.

        Returns:
            Float64 numpy array of shape ``(len(texts), self.dimension)``.
            Each row is the L2-normalised embedding of the corresponding text.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float64)

        matrix = np.zeros((len(texts), self.dimension), dtype=np.float64)
        for i, text in enumerate(texts):
            matrix[i] = self.embed(text)
        return matrix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase and split *text* on non-alphabetic character runs.

        Args:
            text: Raw input string.

        Returns:
            List of alphabetic tokens (empty strings filtered out).
        """
        lowered = text.lower()
        parts = self._TOKEN_RE.split(lowered)
        return [t for t in parts if t]

    def _hash_token(self, token: str) -> int:
        """Map a token to a bucket index in ``[0, dimension)``.

        Uses Python's built-in ``hash()`` with a fixed seed-like offset so
        that the same token always maps to the same bucket within a process.
        We XOR with a large prime to improve bucket distribution.

        Args:
            token: Single alphabetic token string.

        Returns:
            Integer bucket index in ``[0, self.dimension)``.
        """
        # Python's hash() is randomised per-process in CPython >= 3.3 by
        # default (PYTHONHASHSEED).  We use a deterministic FNV-1a variant
        # so that embeddings are reproducible across runs.
        h = 2166136261  # FNV offset basis (32-bit)
        for char in token:
            h ^= ord(char)
            h = (h * 16777619) & 0xFFFFFFFF  # FNV prime, keep 32-bit
        return h % self.dimension

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        """Return the L2-normalised form of *vector*.

        If the vector norm is zero (all-zero input), the zero vector is
        returned unchanged to avoid division by zero.

        Args:
            vector: 1-D float array.

        Returns:
            Unit-norm float array of the same shape.
        """
        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            return vector
        return vector / norm
