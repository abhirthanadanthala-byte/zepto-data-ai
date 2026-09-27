from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("\nCreating ChromaDB...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# Delete the old collection if it exists.
# This makes the script safe to rerun.

try:
    client.delete_collection(
        name=COLLECTION_NAME
    )
except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Zepto support policy documents"
    }
)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = []
metadatas = []
ids = []


doc_files = sorted(
    DOCS_DIR.glob("doc_*.txt")
)


if len(doc_files) != 8:

    raise ValueError(
        f"Expected exactly 8 documents, "
        f"but found {len(doc_files)}."
    )


for doc_file in doc_files:

    text = doc_file.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            f"{doc_file.name} is empty."
        )

    documents.append(text)

    metadatas.append(
        {
            "source": doc_file.name
        }
    )

    ids.append(
        doc_file.stem
    )


print(
    f"\nLoaded {len(documents)} documents."
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    documents,
    normalize_embeddings=True
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# STORE IN CHROMADB
# ============================================================

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


print("\nDocuments stored in ChromaDB.")


# ============================================================
# VERIFY COLLECTION
# ============================================================

count = collection.count()

print(
    "Documents in ChromaDB:",
    count
)


# ============================================================
# TEST RETRIEVAL
# ============================================================

test_query = (
    "How long do I have to report damaged items?"
)


query_embedding = model.encode(
    [test_query],
    normalize_embeddings=True
)


results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)


print("\n" + "=" * 70)
print("TEST RETRIEVAL")
print("=" * 70)

print("Query:")
print(test_query)

print("\nRetrieved documents:")

for i, document in enumerate(
    results["documents"][0]
):

    source = results["metadatas"][0][i]["source"]

    distance = results["distances"][0][i]

    print("\nSource:", source)

    print(
        "Distance:",
        round(distance, 4)
    )

    print(
        "Content:",
        document[:300]
    )


print("\n" + "=" * 70)
print("INGESTION COMPLETED SUCCESSFULLY")
print("=" * 70)