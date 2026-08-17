from flask import Flask, render_template, request, jsonify

from rag_service import ask_question


# ============================================================
# Flask Configuration
# ============================================================

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)


# ============================================================
# Home Page
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# RAG API
# ============================================================

@app.route("/api/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON request."
            }), 400

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400

        # ----------------------------------------------------
        # Send question to RAG service
        # ----------------------------------------------------

        result = ask_question(
            question
        )

        # ----------------------------------------------------
        # Return RAG response
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "answer": result.get(
                "answer",
                ""
            ),
            "sources": result.get(
                "sources",
                []
            ),
            "confidence": result.get(
                "confidence",
                "low"
            ),
            "grounded": result.get(
                "grounded",
                False
            ),
            "retrieval_distance": result.get(
                "retrieval_distance"
            ),
            "image_url": result.get(
                "image_url"
            ),
            "groq_called": result.get(
                "groq_called",
                False
            )
        })

    except Exception as e:

        print(
            f"API Error: {e}"
        )

        return jsonify({
            "success": False,
            "error": (
                "An internal error occurred "
                "while processing your question."
            )
        }), 500


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )