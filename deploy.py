from flask import Flask, render_template, request
import pickle

# Load model
model = pickle.load(open('saved_model.sav', 'rb'))

# Label map for decoding model output
label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    result = ''
    return render_template('index.html', result=result)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form values from the HTML form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Predict using the model
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

    # Map numeric prediction to class name
    result = label_map.get(prediction, "Unknown")

    return render_template('index.html', result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
