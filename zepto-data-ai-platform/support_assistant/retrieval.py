from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve_documents(query: str, top_k: int = 3):
    """
    Retrieve the most relevant Zepto policy documents
    from ChromaDB using semantic similarity.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved.append(
            {
                "document": document,
                "source": metadata["source"],
                "distance": float(distance)
            }
        )

    return retrieved


if __name__ == "__main__":

    query = "How long do I have to report damaged items?"

    print("\nQuery:")
    print(query)

    results = retrieve_documents(query, top_k=3)

    print("\nRetrieved documents:")

    for result in results:
        print("\nSource:", result["source"])
        print("Distance:", round(result["distance"], 4))
        print("Content:", result["document"][:300])