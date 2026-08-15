import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==================================================
# LOAD ENVIRONMENT
# ==================================================

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ==================================================
# EMBEDDINGS
# ==================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# ==================================================
# LOAD FAISS
# ==================================================

print("Loading FAISS database...")

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Ready!")

# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are a Telecom 3GPP Standards Expert.

Instructions:

1. Answer ONLY using the provided context.
2. Never use outside knowledge.
3. If the answer is not found, say:

"The provided 3GPP documentation does not contain sufficient information to answer this question."

4. Do NOT mention source files or page numbers.
5. Prefer official definitions when available.
6. For network functions (AMF, SMF, UPF, PCF, NSSF, AUSF):
   - First give definition.
   - Then list responsibilities/functions.
7. For procedures:
   - Explain objective.
   - Explain major steps.
8. Be concise but technically correct.
"""

# ==================================================
# CHAT LOOP
# ==================================================

while True:

    question = input("\nAsk Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    try:

        print("Searching documents...")

        # -------------------------
        # Similarity Search
        # -------------------------

        results = db.similarity_search_with_score(
            question,
            k=15
        )

        docs = []

        for doc, score in results:

            # lower score = better match
            if score < 0.90:
                docs.append(doc)

        # -------------------------
        # Remove Duplicates
        # -------------------------

        unique_docs = []
        seen = set()

        for doc in docs:

            text = doc.page_content[:300]

            if text not in seen:
                seen.add(text)
                unique_docs.append(doc)

        docs = unique_docs[:8]

        print("\nRetrieved Documents:")

        for doc in docs:
            print(
                f"{doc.metadata.get('source')} | "
                f"Page {doc.metadata.get('page')}"
            )

        if len(docs) == 0:
            print("No relevant documents found.")
            continue

        # -------------------------
        # Build Context
        # -------------------------

        context = ""

        for doc in docs:

            context += f"""

{doc.page_content}

================================================

"""

        print("\nCalling Groq...")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""
Question:

{question}

Context:

{context}
"""
                }
            ]
        )

        answer = response.choices[0].message.content

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)

    except Exception as e:

        print("\nERROR:")
        print(str(e))