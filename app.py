from flask import Flask, request, jsonify
from preprocessing import preprocess_image_bytes
from model import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    uploaded_files = request.files.getlist("file") or request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "No image(s) uploaded"}), 400

    results = []
    for file in uploaded_files:
        image_bytes = file.read()
        image_array = preprocess_image_bytes(image_bytes)
        predictions = predict(image_array)
        results.append({
            "filename": file.filename,
            "predictions": predictions
        })

    return jsonify(results[0] if len(results) == 1 else results)

if __name__ == "__main__":
    app.run(debug=True)

