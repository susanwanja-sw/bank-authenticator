from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize Flask
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict from form
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])[0]
    result = "REAL banknote 🟢" if prediction == 0 else "FAKE banknote 🔴"
    return render_template('index.html', prediction_text=f'This banknote is {result}')

# Optional: Predict via API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])[0]
    return jsonify(int(prediction))

if __name__ == "__main__":
    app.run(debug=True)
