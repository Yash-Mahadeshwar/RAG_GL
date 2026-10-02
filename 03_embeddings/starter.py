"""Step 03 - Embeddings and similarity.

Turn sentences into vectors, then show that sentences with similar *meaning*
end up close together, even when they share no words.

Run from the repo root:
    python 03_embeddings/starter.py
    python 03_embeddings/starter.py "my laptop won't turn on"

Stuck? Peek at solutions/03_embeddings.py
"""

# Default: OpenAI. To use Mistral/OpenRouter/other, change these two lines:
#   client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=...)
#   model = "mistral-small-latest"  # or whichever model
# (Embeddings need an embedding model too, e.g. EMBEDDING_MODEL = "mistral-embed".)

import os
import sys
import traceback
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL = "text-embedding-3-small"

# Three pairs of paraphrases. Each pair means the same thing with different words.
SENTENCES = [
    "The cat sat on the mat.",
    "A kitten was resting on the rug.",
    "Stock markets fell sharply on Monday.",
    "Investors lost money as share prices dropped.",
    "How do I reset my password?",
    "I forgot my login credentials.",
]
DEFAULT_QUERY = "My account won't let me sign in"


def make_client() -> OpenAI:
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and paste your key in.")
    return OpenAI()


def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    # TODO 1: Turn a list of texts into a 2-D numpy array, one row per text.
    #   - response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    #     (send the whole list in ONE call - much faster than looping)
    #   - response.data is a list; each item has an .embedding (a list of floats)
    #   - Return np.array([...those embeddings...])  -> shape (len(texts), 1536)
    raise NotImplementedError("TODO 1: call client.embeddings.create and return a numpy array")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # TODO 2: Cosine similarity = dot(a, b) / (|a| * |b|)
    #   - np.dot(a, b) for the dot product
    #   - np.linalg.norm(a) for the length |a|
    #   - Wrap the result in float() so it prints nicely.
    # Cosine compares DIRECTION only: 1.0 = same meaning, ~0 = unrelated.
    raise NotImplementedError("TODO 2: implement cosine similarity with numpy")


def most_similar(query_vec: np.ndarray, matrix: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    # TODO 3: Return the k rows of `matrix` most similar to `query_vec`,
    #         as a list of (row_index, score) tuples, best first.
    #   - Score every row with cosine_similarity(query_vec, row)
    #   - Sort by score, highest first (sorted(..., key=..., reverse=True))
    #   - Keep the first k
    # This is retrieval in miniature. A vector database does exactly this, faster.
    raise NotImplementedError("TODO 3: score every row and return the top k")


def print_similarity_matrix(vectors: np.ndarray) -> None:
    labels = [f"S{i}" for i in range(len(vectors))]
    print("      " + "  ".join(f"{label:>5}" for label in labels))
    for i, row_label in enumerate(labels):
        cells = "  ".join(f"{cosine_similarity(vectors[i], vectors[j]):5.2f}" for j in range(len(vectors)))
        print(f"{row_label:>5} {cells}")


def main() -> None:
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    client = make_client()

    vectors = embed(client, SENTENCES)
    print(f"Embedded {len(SENTENCES)} sentences -> matrix of shape {vectors.shape}")
    print(f"First 5 numbers of S0: {np.round(vectors[0][:5], 4)}\n")

    for i, sentence in enumerate(SENTENCES):
        print(f"  S{i}: {sentence}")
    print("\nCosine similarity matrix (look for the bright pairs: S0/S1, S2/S3, S4/S5):")
    print_similarity_matrix(vectors)

    query_vec = embed(client, [query])[0]
    print(f'\nQuery: "{query}"')
    for idx, score in most_similar(query_vec, vectors, k=3):
        print(f"  {score:.3f}  S{idx}: {SENTENCES[idx]}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        sys.exit(f"\n[not done yet] line {line}: {exc}\nOpen this file, find that TODO and fill it in.")
