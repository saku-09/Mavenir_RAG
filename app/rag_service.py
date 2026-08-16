import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_DB_PATH = str(PROJECT_ROOT / "vector_db")
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = PROJECT_ROOT / "static" / "generated"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Lower FAISS L2 distance = better retrieval
MAX_DISTANCE = 0.75

REFUSAL_MESSAGE = (
    "The provided 3GPP documentation does not contain sufficient "
    "information to answer this question."
)

# Create generated image directory
IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )

client = Groq(
    api_key=api_key
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ============================================================
# FAISS DATABASE
# ============================================================

print("Loading FAISS database...")

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("RAG service ready!")


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a Telecom 3GPP Standards Assistant.

STRICT RULES:

1. Use ONLY the retrieved 3GPP context supplied by the application.
2. Never use external knowledge.
3. Never invent information.
4. Never infer facts that are not clearly supported by the context.
5. If the context does not clearly support the answer, reply exactly:

{REFUSAL_MESSAGE}

6. Every factual statement must be supported by the retrieved context.
7. Never invent a 3GPP document, section, figure, or page number.
8. Do not claim that a diagram exists unless the supplied context supports it.
9. Keep answers concise and technical.
10. Do not answer questions using knowledge that is absent from the context.

For definition questions use:

Definition:
<definition>

Key Functions:
- item 1
- item 2
- item 3

Only include information explicitly supported by the retrieved context.
"""


# ============================================================
# VISUAL QUERY DETECTION
# ============================================================

def is_visual_query(question: str) -> bool:
    """
    Determines whether the user wants a diagram,
    architecture, image, topology, or visual representation.
    """

    q = question.lower().strip()

    visual_terms = [
        "show me",
        "show",
        "display",
        "diagram",
        "architecture",
        "architectural diagram",
        "network architecture",
        "core architecture",
        "5g core architecture",
        "5g system architecture",
        "topology",
        "flow diagram",
        "call flow",
        "signaling flow",
        "procedure flow",
        "figure",
        "image",
        "picture",
        "visual",
        "illustration",
    ]

    return any(
        term in q
        for term in visual_terms
    )


# ============================================================
# GENERAL 5G CORE ARCHITECTURE QUERY DETECTION
# ============================================================

def is_general_5g_core_architecture_query(
    question: str
) -> bool:
    """
    Detects a request for the general 5G Core / 5G System
    architecture.

    Specialized requests such as:
    - non-3GPP architecture
    - trusted architecture
    - untrusted architecture
    - roaming architecture

    are intentionally excluded.
    """

    q = question.lower()

    specialized_terms = [
        "non-3gpp",
        "non 3gpp",
        "trusted non-3gpp",
        "untrusted non-3gpp",
        "roaming architecture",
        "trusted architecture",
        "untrusted architecture",
        "access architecture",
    ]

    if any(
        term in q
        for term in specialized_terms
    ):
        return False

    general_terms = [
        "5g core architecture",
        "5g system architecture",
        "general 5g architecture",
        "general 5g core",
        "5g core network architecture",
        "core network architecture",
        "show me the 5g core",
        "show me 5g core",
    ]

    return any(
        term in q
        for term in general_terms
    )


# ============================================================
# VISUAL KEYWORDS
# ============================================================

def get_visual_keywords(question: str):
    """
    Returns keywords used when selecting a PDF page
    for a visual answer.
    """

    q = question.lower()

    # --------------------------------------------------------
    # General 5G Core architecture
    # --------------------------------------------------------

    if is_general_5g_core_architecture_query(
        question
    ):

        return [
            "4.2.3",
            "non-roaming reference architecture",
            "non-roaming 5g system architecture",
            "figure 4.2.3-1",
            "5g system architecture",
            "5g core network",
            "network functions",
            "amf",
            "smf",
            "upf",
            "udm",
            "pcf",
            "ausf",
            "nrf",
            "nssf",
        ]

    # --------------------------------------------------------
    # Non-3GPP architecture
    # --------------------------------------------------------

    if (
        "non-3gpp" in q
        or "non 3gpp" in q
    ):

        return [
            "4.2.8",
            "non-3gpp",
            "untrusted non-3gpp",
            "trusted non-3gpp",
            "n3iwf",
            "tngf",
        ]

    # --------------------------------------------------------
    # AMF
    # --------------------------------------------------------

    if "amf" in q:

        return [
            "amf",
            "access and mobility management function",
            "5g core",
            "network function",
        ]

    # --------------------------------------------------------
    # SMF
    # --------------------------------------------------------

    if "smf" in q:

        return [
            "smf",
            "session management function",
            "pdu session",
            "5g core",
        ]

    # --------------------------------------------------------
    # UPF
    # --------------------------------------------------------

    if "upf" in q:

        return [
            "upf",
            "user plane function",
            "pdu session",
            "user plane",
        ]

    # --------------------------------------------------------
    # NG-RAN
    # --------------------------------------------------------

    if (
        "ng-ran" in q
        or "ran architecture" in q
        or "gnb" in q
    ):

        return [
            "ng-ran",
            "gnb",
            "radio access network",
            "ng-ran architecture",
        ]

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    if "registration" in q:

        return [
            "registration procedure",
            "registration",
            "registration request",
            "registration accept",
        ]

    # --------------------------------------------------------
    # Network slicing
    # --------------------------------------------------------

    if "network slicing" in q:

        return [
            "network slicing",
            "network slice",
            "nssf",
            "slice",
        ]

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    return [
        word
        for word in re.findall(
            r"[a-zA-Z0-9-]+",
            q
        )
        if len(word) > 3
    ]


# ============================================================
# PDF PAGE TEXT
# ============================================================

def get_page_text(
    page
) -> str:

    try:
        return page.get_text(
            "text"
        ).lower()

    except Exception:
        return ""


# ============================================================
# PDF VISUAL SCORE
# ============================================================

def score_visual_page(
    page,
    keywords,
    general_architecture=False
):
    """
    Score one PDF page.

    Higher score = better visual candidate.
    """

    text = get_page_text(
        page
    )

    score = 0

    # --------------------------------------------------------
    # Keyword matching
    # --------------------------------------------------------

    for keyword in keywords:

        keyword_lower = keyword.lower()

        if keyword_lower in text:

            if keyword_lower in {
                "4.2.3",
                "figure 4.2.3-1",
                "non-roaming reference architecture",
                "non-roaming 5g system architecture",
            }:

                score += 35

            elif keyword_lower in {
                "5g system architecture",
                "5g core network",
            }:

                score += 20

            else:

                score += 8

    # --------------------------------------------------------
    # Vector drawings
    # --------------------------------------------------------

    try:

        drawing_count = len(
            page.get_drawings()
        )

    except Exception:

        drawing_count = 0

    # --------------------------------------------------------
    # Embedded images
    # --------------------------------------------------------

    try:

        image_count = len(
            page.get_images(
                full=True
            )
        )

    except Exception:

        image_count = 0

    # Architecture diagrams often have vector drawings
    score += min(
        drawing_count,
        30
    ) * 1.5

    score += min(
        image_count,
        10
    ) * 4

    # --------------------------------------------------------
    # General 5G architecture preference
    # --------------------------------------------------------

    if general_architecture:

        # Strong preference for official general
        # 5G architecture section.
        if "4.2.3" in text:

            score += 100

        if "non-roaming reference architecture" in text:

            score += 100

        if "figure 4.2.3-1" in text:

            score += 120

        if "non-roaming 5g system architecture" in text:

            score += 100

        # Prefer pages containing core NFs
        nf_terms = [
            "amf",
            "smf",
            "upf",
            "udm",
            "pcf",
            "ausf",
            "nrf",
            "nssf",
        ]

        nf_count = sum(
            1
            for term in nf_terms
            if term in text
        )

        score += nf_count * 8

        # ----------------------------------------------
        # Strong penalties for specialized architecture
        # ----------------------------------------------

        if "non-3gpp" in text:

            score -= 100

        if "trusted non-3gpp" in text:

            score -= 100

        if "untrusted non-3gpp" in text:

            score -= 100

        if "n3iwf" in text:

            score -= 60

        if "tngf" in text:

            score -= 60

        if "roaming architecture" in text:

            score -= 50

    return score


# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_pdf_page(
    source_name,
    page_index,
    output_suffix="page"
):
    """
    Render one PDF page to PNG.

    page_index is zero-based for PyMuPDF.
    """

    if not source_name:
        return None

    pdf_path = (
        DATA_DIR /
        Path(source_name).name
    )

    if not pdf_path.exists():

        print(
            f"PDF not found: {pdf_path}"
        )

        return None

    try:

        page_index = int(
            page_index
        )

    except (
        TypeError,
        ValueError
    ):

        return None

    try:

        document = fitz.open(
            str(pdf_path)
        )

        if (
            page_index < 0
            or page_index >= len(document)
        ):

            document.close()

            return None

        page = document[
            page_index
        ]

        # Good quality for web display
        matrix = fitz.Matrix(
            1.5,
            1.5
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            Path(source_name).stem
        )

        image_name = (
            f"{safe_name}_"
            f"{output_suffix}_"
            f"{page_index + 1}.png"
        )

        image_path = (
            IMAGE_DIR /
            image_name
        )

        pixmap.save(
            str(image_path)
        )

        document.close()

        return (
            f"/static/generated/"
            f"{image_name}"
        )

    except Exception as exc:

        print(
            f"PDF rendering error: {exc}"
        )

        return None


# ============================================================
# FIND BEST VISUAL PAGE
# ============================================================

def find_best_visual_page(
    question,
    preferred_source=None,
    preferred_page=None
):
    """
    Find the most relevant diagram page.

    General 5G Core architecture gets special treatment:
    TS 23.501 -> Section 4.2.3 -> Figure 4.2.3-1.
    """

    general_architecture = (
        is_general_5g_core_architecture_query(
            question
        )
    )

    keywords = get_visual_keywords(
        question
    )

    candidate_files = []

    # --------------------------------------------------------
    # For the general 5G Core architecture,
    # always prioritize TS 23.501.
    # --------------------------------------------------------

    if general_architecture:

        architecture_pdf = (
            DATA_DIR /
            "ts_123501v171500p.pdf"
        )

        if architecture_pdf.exists():

            candidate_files.append(
                architecture_pdf
            )

    # --------------------------------------------------------
    # Preferred source from FAISS
    # --------------------------------------------------------

    if preferred_source:

        preferred_path = (
            DATA_DIR /
            Path(preferred_source).name
        )

        if preferred_path.exists():

            if preferred_path not in candidate_files:

                candidate_files.append(
                    preferred_path
                )

    # --------------------------------------------------------
    # Remaining PDFs
    # --------------------------------------------------------

    for pdf_file in DATA_DIR.glob(
        "*.pdf"
    ):

        if pdf_file not in candidate_files:

            candidate_files.append(
                pdf_file
            )

    best = None

    # --------------------------------------------------------
    # Scan documents
    # --------------------------------------------------------

    for pdf_path in candidate_files:

        try:

            document = fitz.open(
                str(pdf_path)
            )

        except Exception as exc:

            print(
                f"Could not open "
                f"{pdf_path}: {exc}"
            )

            continue

        for page_index in range(
            len(document)
        ):

            page = document[
                page_index
            ]

            page_score = score_visual_page(
                page=page,
                keywords=keywords,
                general_architecture=general_architecture,
            )

            # ------------------------------------------------
            # FAISS page bonus
            # ------------------------------------------------

            if (
                preferred_source
                and Path(
                    preferred_source
                ).name == pdf_path.name
                and preferred_page is not None
            ):

                try:

                    if (
                        page_index
                        == int(
                            preferred_page
                        )
                    ):

                        page_score += 5

                except (
                    TypeError,
                    ValueError
                ):

                    pass

            # ------------------------------------------------
            # Best page
            # ------------------------------------------------

            if (
                best is None
                or page_score > best["score"]
            ):

                best = {
                    "source": pdf_path.name,
                    "page_index": page_index,
                    "score": page_score,
                }

        document.close()

    return best


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(
    filtered_docs
):

    context_parts = []

    for doc, score in filtered_docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{doc.page_content}

====================
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def ask_question(
    question: str
):

    question = question.strip()

    # --------------------------------------------------------
    # Empty question
    # --------------------------------------------------------

    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "confidence": "low",
            "grounded": False,
            "retrieval_distance": None,
            "image_url": None,
            "groq_called": False,
        }

    # --------------------------------------------------------
    # FAISS retrieval
    # --------------------------------------------------------

    print(
        f"\nQuestion: {question}"
    )

    print(
        "Searching 3GPP documents..."
    )

    results = db.similarity_search_with_score(
        question,
        k=8
    )

    # --------------------------------------------------------
    # No retrieval
    # --------------------------------------------------------

    if not results:

        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "confidence": "low",
            "grounded": False,
            "retrieval_distance": None,
            "image_url": None,
            "groq_called": False,
        }

    # --------------------------------------------------------
    # Best distance
    # --------------------------------------------------------

    best_score = min(
        float(score)
        for _, score in results
    )

    print(
        f"Best retrieval distance: "
        f"{best_score:.4f}"
    )

    # --------------------------------------------------------
    # Evidence gate
    # --------------------------------------------------------

    if best_score > MAX_DISTANCE:

        print(
            "Evidence insufficient."
        )

        print(
            "Groq will NOT be called."
        )

        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "confidence": "low",
            "grounded": False,
            "retrieval_distance": round(
                best_score,
                4
            ),
            "image_url": None,
            "groq_called": False,
        }

    # --------------------------------------------------------
    # Remove duplicate source pages
    # --------------------------------------------------------

    filtered_docs = []

    seen = set()

    for doc, score in results:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        key = (
            source,
            page
        )

        if key not in seen:

            seen.add(
                key
            )

            filtered_docs.append(
                (
                    doc,
                    float(score)
                )
            )

    if not filtered_docs:

        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "confidence": "low",
            "grounded": False,
            "retrieval_distance": round(
                best_score,
                4
            ),
            "image_url": None,
            "groq_called": False,
        }

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        filtered_docs
    )

    # Keep context manageable
    context = context[:12000]

    # --------------------------------------------------------
    # Groq generation
    # --------------------------------------------------------

    print(
        "Calling Groq..."
    )

    try:

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

        answer = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:

        print(
            f"Groq error: {exc}"
        )

        return {
            "answer": (
                "The answer could not be generated "
                "because the language model request failed."
            ),
            "sources": [],
            "confidence": "low",
            "grounded": False,
            "retrieval_distance": round(
                best_score,
                4
            ),
            "image_url": None,
            "groq_called": True,
        }

    if (
        not answer
        or not answer.strip()
    ):

        answer = REFUSAL_MESSAGE

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if best_score <= 0.50:

        confidence = "high"

    elif best_score <= 0.65:

        confidence = "medium"

    else:

        confidence = "low"

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = []

    for doc, score in filtered_docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        sources.append({
            "document": source,
            "page": page,
            "distance": round(
                score,
                4
            ),
            "image_url": None,
        })

    # --------------------------------------------------------
    # Visual retrieval
    # --------------------------------------------------------

    image_url = None

    if is_visual_query(
        question
    ):

        print(
            "Visual query detected."
        )

        top_doc, top_score = (
            filtered_docs[0]
        )

        preferred_source = (
            top_doc.metadata.get(
                "source"
            )
        )

        preferred_page = (
            top_doc.metadata.get(
                "page"
            )
        )

        visual_page = (
            find_best_visual_page(
                question=question,
                preferred_source=preferred_source,
                preferred_page=preferred_page,
            )
        )

        if visual_page:

            print(
                "\nBest visual page:"
            )

            print(
                f"Document: "
                f"{visual_page['source']}"
            )

            print(
                f"PDF page index: "
                f"{visual_page['page_index']}"
            )

            print(
                f"Visual score: "
                f"{visual_page['score']}"
            )

            image_url = render_pdf_page(
                source_name=visual_page[
                    "source"
                ],
                page_index=visual_page[
                    "page_index"
                ],
                output_suffix="diagram",
            )

            if image_url:

                # --------------------------------------------
                # Find matching source
                # --------------------------------------------

                matched = False

                for source_item in sources:

                    if (
                        source_item[
                            "document"
                        ]
                        == visual_page[
                            "source"
                        ]
                    ):

                        source_item[
                            "image_url"
                        ] = image_url

                        source_item[
                            "visual_page_index"
                        ] = visual_page[
                            "page_index"
                        ]

                        matched = True

                        break

                # --------------------------------------------
                # Add visual page as an additional source
                # --------------------------------------------

                if not matched:

                    sources.append({
                        "document":
                            visual_page[
                                "source"
                            ],

                        "page":
                            visual_page[
                                "page_index"
                            ],

                        "distance":
                            None,

                        "image_url":
                            image_url,

                        "visual_page":
                            True,
                    })

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "grounded": True,
        "retrieval_distance": round(
            best_score,
            4
        ),
        "image_url": image_url,
        "groq_called": True,
    }