"""Retrieval-augmented generation (RAG).

Demonstrates:
- Offline: chunking documents, embedding, and indexing
- Online: embedding a query, retrieving relevant chunks, and generating
  a grounded answer
- ``@Tool.define`` to expose retrieval as a tool the LLM can call
- Separation of indexing (plain Python) from generation (``@Template.define``)
"""

import argparse
import dataclasses

import litellm
import numpy as np

from effectful.handlers.llm import Agent, Template, Tool

# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class VectorIndex:
    """Simple in-memory vector index using L2 distance."""

    model: str
    chunks: list[str] = dataclasses.field(default_factory=list)
    embeddings: list[np.ndarray] = dataclasses.field(default_factory=list)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get an embedding vector for the given text using litellm."""
        response = litellm.embedding(model=self.model, input=text)
        return np.array(response.data[0]["embedding"], dtype=np.float32)

    def add(self, text: str) -> None:
        """Add a text chunk to the index."""
        self.chunks.append(text)
        self.embeddings.append(self.get_embedding(text))

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Return the top-k most similar chunks to the query."""
        if not self.embeddings:
            return []
        query_emb = self.get_embedding(query)
        distances = [float(((emb - query_emb) ** 2).sum()) for emb in self.embeddings]
        indices = sorted(range(len(distances)), key=lambda i: distances[i])
        return [self.chunks[i] for i in indices[:top_k]]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

DOCUMENTS = [
    """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars
    in Paris, France. It is named after the engineer Gustave Eiffel, whose
    company designed and built the tower from 1887 to 1889 as the centerpiece
    of the 1889 World's Fair. Although initially criticized by some of France's
    leading artists and intellectuals, the tower has become a global icon of
    France and one of the most recognizable structures in the world. The tower
    is 330 metres tall, about the same height as an 81-storey building, and
    is the tallest structure in Paris. It was the first structure in the world
    to reach a height of 300 metres.""",
    """The Great Wall of China is a series of fortifications that were built
    across the historical northern borders of ancient Chinese states and
    Imperial China as protection against various nomadic groups. The total
    length of all sections ever built is more than 20,000 km. Several walls
    were built from as early as the 7th century BC, with selective stretches
    later joined together by Qin Shi Huang, the first emperor of China. The
    best-preserved sections of the wall date from the Ming dynasty
    (1368-1644). The wall's purpose was defensive, and it featured
    watchtowers, troop barracks, and signaling capabilities.""",
    """The Colosseum, also known as the Flavian Amphitheatre, is an oval
    amphitheatre in the centre of the city of Rome, Italy. It is the largest
    ancient amphitheatre ever built, and is still the largest standing
    amphitheatre in the world, despite its age. Construction began under
    the emperor Vespasian in AD 72 and was completed in AD 80 under his
    successor and heir, Titus. The Colosseum could hold an estimated 50,000
    to 80,000 spectators at various points in its history, and was used for
    gladiatorial contests and public spectacles including animal hunts,
    executions, re-enactments of famous battles, and dramas.""",
]

# ---------------------------------------------------------------------------
# Build the index (offline phase)
# ---------------------------------------------------------------------------


def build_index(documents: list[str], embedding_model: str) -> VectorIndex:
    """Chunk and index a collection of documents."""
    index = VectorIndex(model=embedding_model)
    for doc in documents:
        for chunk in chunk_text(doc, chunk_size=60, overlap=15):
            index.add(chunk)
    print(f"Indexed {len(index.chunks)} chunks from {len(documents)} documents")
    return index


# ---------------------------------------------------------------------------
# RAG agent (online phase): the `retrieve` tool and `answer_question` template
# share one instance, so the tool is auto-captured from lexical scope.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RAGAgent(Agent):
    """Answers a question grounded in the vector index via a retrieval tool."""

    index: VectorIndex

    @Tool.define
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return the top-k most similar chunks to the query."""
        return self.index.search(query, top_k)

    @Template.define
    def answer_question(self, question: str) -> str:
        """You are a helpful assistant. Answer the user's question using ONLY
        information retrieved from the knowledge base via the retrieve tool.

        If the retrieved information doesn't contain the answer, say so.
        Always cite which document your information comes from.

        Question: {question}
        """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="lm_studio/text-embedding-embeddinggemma-300m-qat",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        metavar="QUESTION",
        default=[
            "How tall is the Eiffel Tower?",
            "When was the Great Wall of China built?",
            "How many spectators could the Colosseum hold?",
        ],
        help="Questions to answer against the indexed documents",
    )
    args = parser.parse_args()

    # Offline: build the index once.
    index = build_index(DOCUMENTS, embedding_model=args.embedding_model)

    # Online: answer each question with a fresh (stateless) agent over that index.
    for question in args.questions:
        print(f"\nQ: {question}")
        answer = RAGAgent(index=index).answer_question(question)
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
