from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Convert categorical yes/no inputs to 1/0
        mainroad = 1 if data.get("mainroad") == "yes" else 0
        guestroom = 1 if data.get("guestroom") == "yes" else 0
        basement = 1 if data.get("basement") == "yes" else 0
        hotwaterheating = 1 if data.get("hotwaterheating") == "yes" else 0
        airconditioning = 1 if data.get("airconditioning") == "yes" else 0
        prefarea = 1 if data.get("prefarea") == "yes" else 0

        # Furnishing status encoding
        furnishingstatus_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
        furnishingstatus = furnishingstatus_map.get(data.get("furnishingstatus"), 0)

        # Arrange inputs as DataFrame (must match training columns order!)
        features = pd.DataFrame([[
            float(data["area"]),
            int(data["bedrooms"]),
            int(data["bathrooms"]),
            int(data["stories"]),
            mainroad,
            guestroom,
            basement,
            hotwaterheating,
            airconditioning,
            int(data["parking"]),
            prefarea,
            furnishingstatus
        ]], columns=[
            "area", "bedrooms", "bathrooms", "stories", "mainroad",
            "guestroom", "basement", "hotwaterheating", "airconditioning",
            "parking", "prefarea", "furnishingstatus"
        ])

        # Predict
        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
