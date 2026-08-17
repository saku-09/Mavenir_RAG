# Mavenir 3GPP Standards RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot designed to answer questions from Telecom 3GPP standards documentation.

This project was developed as part of the **Mavenir Graduate Engineer Trainee (GET) technical evaluation**.

The primary objective is to build a technical chatbot that uses **3GPP standards as its knowledge source** and reduces hallucination by validating retrieval evidence before generating an answer.

---

## Project Overview

The system combines:

- 3GPP Telecom standards
- Hugging Face embeddings
- FAISS vector search
- Retrieval distance thresholding
- Groq LLM
- Strict grounding prompts
- Flask REST API
- HTML/CSS/JavaScript frontend
- Source and page tracking
- Confidence and grounding indicators
- Initial document/diagram visualization support

The main design principle is:

> **Retrieve relevant 3GPP evidence first. Generate an answer only when sufficient evidence is available. Otherwise, refuse to answer.**

---

# System Architecture

```text
                         USER
                           |
                           v
                    Web Chat Interface
                           |
                           v
                     Flask API
                           |
                           v
                     User Question
                           |
                           v
                BGE Embedding Model
                           |
                           v
                    FAISS Search
                           |
                           v
                 Retrieval Distance
                      Evaluation
                           |
             +-------------+-------------+
             |                           |
      Sufficient Evidence        Insufficient Evidence
             |                           |
             v                           v
      Retrieve 3GPP Context          REFUSE
             |
             v
       Context Validation
             |
             v
          Groq LLM
             |
             v
       Grounded Answer
             |
             v
         Flask API
             |
             v
        Web Interface

RAG Pipeline

3GPP PDF Documents
        |
        v
Document Processing
        |
        v
Text Chunks
        |
        v
BGE Embeddings
        |
        v
FAISS Vector Database
        |
        v
User Question
        |
        v
Semantic Retrieval
        |
        v
Evidence / Distance Check
        |
        +----------------------+
        |                      |
        | Evidence Sufficient  | Evidence Insufficient
        |                      |
        v                      v
  Retrieved Context          Refusal
        |
        v
      Groq
        |
        v
Grounded Answer