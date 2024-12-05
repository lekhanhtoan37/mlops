from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

model = mlflow.sklearn.load_model("model")

@app.route("/", methods=["POST"])
def predict():
    features = request.json["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
