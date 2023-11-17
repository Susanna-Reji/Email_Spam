from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = Flask(__name__)

# Load your trained model and feature extraction model
model = joblib.load('model.pkl')
feature_extraction = joblib.load('feature_extraction_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        predicted_label = classify_text(input_text)
        
        # Redirect to the result page with the predicted label
        return redirect(url_for('result', predicted_label=predicted_label))

    return render_template('index.html')

@app.route('/result/<predicted_label>', methods=['GET'])
def result(predicted_label):
    return render_template('result.html', predicted_label=predicted_label)

def classify_text(input_text):
    input_data_features = feature_extraction.transform([input_text])
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        return '  HAM MAIL'
    else:
        return ' SPAM MAIL'

if __name__ == '__main__':
    app.run(debug=True)
