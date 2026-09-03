from flask import Flask, render_template, request, jsonify

import models
import llm_router

app = Flask(__name__)

FIELDS = [
    ("Type", "select", ["L", "M", "H"]),
    ("Air temperature [K]", "number", 298.0),
    ("Process temperature [K]", "number", 308.5),
    ("Rotational speed [rpm]", "number", 1500),
    ("Torque [Nm]", "number", 40.0),
    ("Tool wear [min]", "number", 100),
]


def make_result(form):
    result = models.predict_machine_failure(
        air_temp=float(form["Air temperature [K]"]),
        process_temp=float(form["Process temperature [K]"]),
        rpm=float(form["Rotational speed [rpm]"]),
        torque=float(form["Torque [Nm]"]),
        tool_wear=float(form["Tool wear [min]"]),
        product_type=form["Type"],
    )
    result["label"] = "DEFECT / FAILURE LIKELY" if result["predicted_failure"] == 1 else "OK / NO FAILURE PREDICTED"
    result["closest_mode"] = result["closest_specific_failure_mode"]
    result["modes"] = result["failure_modes"]
    return result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=FIELDS, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    result = make_result(request.form)
    return render_template("index.html", fields=FIELDS, result=result, form_values=request.form)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    return jsonify(make_result(data))


@app.route("/ask", methods=["GET"])
def ask_page():
    return render_template("ask.html", question=None, result=None, error=None)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    result, error = None, None
    if question:
        try:
            result = llm_router.answer_question(question)
        except Exception as exc:
            error = str(exc)
    return render_template("ask.html", question=question, result=result, error=error)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    try:
        return jsonify(llm_router.answer_question(question))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


if __name__ == "__main__":
    app.run(debug=True, port=5000)
