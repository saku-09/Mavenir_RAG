import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================
# Load Environment Variables
# =====================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

# =====================================
# Load Embedding Model
# =====================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# =====================================
# Load FAISS Database
# =====================================

print("Loading FAISS database...")

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Ready!")

# =====================================
# System Prompt
# =====================================

SYSTEM_PROMPT = """
You are a Telecom 3GPP Standards Assistant.

STRICT RULES:

1. Use ONLY the provided context.
2. Never use external knowledge.
3. Never invent information.
4. If the answer is not clearly present in the context, reply exactly:

The provided 3GPP documentation does not contain sufficient information to answer this question.

5. For definition questions answer in this format:

Definition:
<definition>

Key Functions:
- item 1
- item 2
- item 3

6. Only include functions that appear in the context.
7. Do not mention source names or page numbers.
8. Keep answers concise and technical.
"""

# =====================================
# Chat Loop
# =====================================

while True:

    question = input("\nAsk Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    try:

        print("Searching documents...")

        docs = db.similarity_search(
            question,
            k=8
        )

        # ---------------------------------
        # Remove duplicate pages
        # ---------------------------------

        filtered_docs = []
        seen = set()

        for doc in docs:

            source = doc.metadata.get("source")
            page = doc.metadata.get("page")

            key = (source, page)

            if key not in seen:
                seen.add(key)
                filtered_docs.append(doc)

        docs = filtered_docs

        if not docs:
            print("No documents found.")
            continue

        print("\nRetrieved Documents:")

        for doc in docs:
            print(
                f"{doc.metadata.get('source')} | Page {doc.metadata.get('page')}"
            )

        # ---------------------------------
        # Build Context
        # ---------------------------------

        context = ""

        for doc in docs:

            context += f"""
{doc.page_content}

====================
"""

        # Limit context size
        context = context[:12000]

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

        if not answer or not answer.strip():
            answer = (
                "The provided 3GPP documentation does not contain "
                "sufficient information to answer this question."
            )

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)

    except Exception as e:

        print("\nERROR:")
        print(str(e))