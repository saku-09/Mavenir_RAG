from flask import Flask, render_template, request, jsonify

from rag_service import ask_question


app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json(silent=True) or {}

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400

        result = ask_question(question)

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )