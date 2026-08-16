import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# Configuration
# ============================================================

VECTOR_DB_PATH = "vector_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# FAISS is returning L2 distance in this index.
# Lower distance = better retrieval.
#
# We will calibrate this value using your actual test results.
MAX_DISTANCE = 0.75

REFUSAL_MESSAGE = (
    "The provided 3GPP documentation does not contain sufficient "
    "information to answer this question."
)


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )

client = Groq(api_key=api_key)


# ============================================================
# Load Embedding Model
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    },
)


# ============================================================
# Load FAISS Vector Database
# ============================================================

print("Loading FAISS database...")

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

print("Ready!")


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = f"""
You are a Telecom 3GPP Standards Assistant.

You answer questions ONLY from the retrieved 3GPP context supplied
by the application.

STRICT RULES:

1. Use ONLY the provided context.
2. Never use outside knowledge.
3. Never use your own background knowledge.
4. Never invent facts.
5. Never infer information that is not clearly supported by the context.
6. If the context does not clearly contain the answer, reply exactly:

{REFUSAL_MESSAGE}

7. Every factual statement must be supported by the retrieved context.
8. Do not invent document names, section numbers, page numbers,
   specifications, procedures, or terminology.
9. Do not include citations or page numbers yourself.
   The application will add verified source information separately.
10. Keep the answer concise and technical.

For definition-style questions use:

Definition:
<definition>

Key Functions:
- item 1
- item 2
- item 3

Only include functions that are clearly supported by the supplied context.
"""


# ============================================================
# Helper Functions
# ============================================================

def print_sources(docs):
    """
    Print verified source information from retrieved document metadata.
    The LLM does not generate these citations.
    """

    print("\nSources:")

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        print(f"[{i}] {source} | Page {page}")


def build_context(filtered_docs):
    """
    Build context from the retrieved documents.
    Source and page metadata are included so the LLM understands
    where the supplied text came from.
    """

    context_parts = []

    for doc, score in filtered_docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{doc.page_content}

====================
"""
        )

    return "\n".join(context_parts)


# ============================================================
# Chat Loop
# ============================================================

while True:

    question = input("\nAsk Question (type exit to quit): ").strip()

    # --------------------------------------------------------
    # Exit Handling
    # --------------------------------------------------------

    if question.lower() in {"exit", "quit", "q"}:
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    try:

        # ----------------------------------------------------
        # Retrieve Documents + Scores
        # ----------------------------------------------------

        print("\nSearching documents...")

        results = db.similarity_search_with_score(
            question,
            k=8
        )

        if not results:

            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)
            print(REFUSAL_MESSAGE)

            continue

        # ----------------------------------------------------
        # Display Retrieval Results
        # ----------------------------------------------------

        print("\nRetrieval Results:")

        for doc, score in results:

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            print(
                f"{source} | "
                f"Page {page} | "
                f"Distance: {score:.4f}"
            )

        # ----------------------------------------------------
        # Evidence Check
        # ----------------------------------------------------

        best_score = min(score for _, score in results)

        print(f"\nBest retrieval distance: {best_score:.4f}")
        print(f"Maximum allowed distance: {MAX_DISTANCE:.4f}")

        # Higher distance means weaker evidence.
        # If the best result is still too far away,
        # do NOT call the LLM.

        if best_score > MAX_DISTANCE:

            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)

            print(REFUSAL_MESSAGE)

            print(
                "\nGroq was NOT called because retrieval "
                "evidence was insufficient."
            )

            continue

        # ----------------------------------------------------
        # Remove Duplicate Pages
        # ----------------------------------------------------

        filtered_docs = []

        seen = set()

        for doc, score in results:

            source = doc.metadata.get("source")
            page = doc.metadata.get("page")

            key = (source, page)

            if key not in seen:

                seen.add(key)

                filtered_docs.append(
                    (doc, score)
                )

        if not filtered_docs:

            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)

            print(REFUSAL_MESSAGE)

            continue

        # ----------------------------------------------------
        # Print Selected Documents
        # ----------------------------------------------------

        print("\nSelected Documents:")

        for doc, score in filtered_docs:

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            print(
                f"{source} | "
                f"Page {page} | "
                f"Distance: {score:.4f}"
            )

        # ----------------------------------------------------
        # Build Context
        # ----------------------------------------------------

        context = build_context(
            filtered_docs
        )

        # Limit context size
        context = context[:12000]

        # ----------------------------------------------------
        # Call Groq
        # ----------------------------------------------------

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

Retrieved 3GPP Context:
{context}
"""
                }
            ]
        )

        answer = response.choices[0].message.content

        # ----------------------------------------------------
        # Empty Response Protection
        # ----------------------------------------------------

        if not answer or not answer.strip():

            answer = REFUSAL_MESSAGE

        # ----------------------------------------------------
        # Final Answer
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)

        print(answer)

        # ----------------------------------------------------
        # Verified Sources
        # ----------------------------------------------------

        print_sources(
            [
                doc
                for doc, score in filtered_docs
            ]
        )

    except Exception as e:

        print("\nERROR:")
        print(str(e))