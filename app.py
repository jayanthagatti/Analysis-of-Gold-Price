from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('gold_model.pkl')  # Ensure this file exists

@app.route('/')
def home():
    return render_template('index.html')  # Loads the form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        spx = float(request.form['SPX'])
        uso = float(request.form['USO'])
        slv = float(request.form['SLV'])
        eurusd = float(request.form['EUR/USD'])  # Match input name exactly

        # Prepare input and predict
        input_data = np.array([[spx, uso, slv, eurusd]])
        prediction = model.predict(input_data)
        

        # Return prediction to the HTML template
        return render_template('index.html', prediction_text=f'Predicted Gold Price: ${prediction[0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
