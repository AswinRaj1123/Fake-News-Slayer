from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('fake_news_model.joblib')

def detect_fake_news(article):
    """
    Predict whether a news article is fake or real using the loaded model.
    
    Parameters:
    article (str): The news article headline to classify.
    
    Returns:
    str: 'The News is Fake' if the headline is fake, 'The News is Real' if the headline is real.
    """
    prediction = model.predict([article])[0]
    return 'The News is Fake' if prediction == 0 else 'The News is Real'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        article = request.form['textInput']
        result = detect_fake_news(article)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
