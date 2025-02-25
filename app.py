from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)  # Create Flask app
model = joblib.load("car_price_predictor.pkl")  # Load saved model

@app.route("/", methods=["GET"])
def home():
    return "Flask App is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get JSON data from request
    print("Received Data:", data)  # Debugging
    if "km_driven" not in data:
        return jsonify({"error": "Missing km_driven"}), 400

    km_driven = np.array([[data["km_driven"]]])  # Convert to 2D array
    predicted_price = model.predict(km_driven)[0]  # Get prediction
    return jsonify({"predicted_price": predicted_price})  # Return result as JSON

if __name__ == "__main__":
    app.run(debug=True)  # Run API
