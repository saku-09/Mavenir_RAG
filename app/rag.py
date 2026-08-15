from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

query = input("Ask Question: ")

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

docs = retriever.invoke(query)

for i, doc in enumerate(docs, start=1):
    print("\n" + "=" * 80)
    print(f"RESULT {i}")

    print(
        f"\nSource: {doc.metadata.get('source')} | "
        f"Page: {doc.metadata.get('page')}"
    )

    print("\nContent:\n")
    print(doc.page_content[:1000])