import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
true_news = pd.read_csv('true.csv')
fake_news = pd.read_csv('fake.csv')

# Add a column to indicate whether the news is true (1) or fake (0)
true_news['label'] = 1
fake_news['label'] = 0

# Combine the datasets
data = pd.concat([true_news, fake_news])

# Select the columns to use
X = data['headline']
y = data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and LogisticRegression
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(model, 'fake_news_model.joblib')

def predict_news(headline):
    """
    Predict whether a news headline is true or fake.
    
    Parameters:
    headline (str): The news headline to classify.
    
    Returns:
    str: 'True' if the headline is true, 'Fake' if the headline is fake.
    """
    prediction = model.predict([headline])[0]
    return 'True' if prediction == 1 else 'Fake'

# Ask for the headline input from the user
headline = input("Enter the news headline: ")
print(f'The headline "{headline}" is {predict_news(headline)}.')
