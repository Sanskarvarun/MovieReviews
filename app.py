from flask import Flask, request, jsonify
import pickle
from nltk import pos_tag
from nltk.corpus import wordnet
import string

app = Flask(__name__)

# Load the pickled model and other necessary components
with open('naive_bayes_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('lemmatizer.pkl', 'rb') as f:
    lemmatizer = pickle.load(f)

with open('stops.pkl', 'rb') as f:
    stops = pickle.load(f)

# Function to get simple POS


def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function to clean and preprocess the review text


def clean_review(words):
    output_words = []
    for w in words:
        if w.lower() not in stops:
            pos = pos_tag([w])
            clean_word = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return output_words

# Function to extract features from the review


def get_feature_dict(words):
    words_set = set(words)
    current_features = {}
    for w in features:
        current_features[w] = (w in words_set)
    return current_features

# Define a route to classify a review


@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the request
    review = request.json.get('review', '')

    # Debugging: print the input review
    print(f"Received review: {review}")

    # Clean and preprocess the review
    words = review.split()  # Tokenize the review text
    cleaned_review = clean_review(words)

    # Debugging: print the cleaned review
    print(f"Cleaned review: {cleaned_review}")

    # Extract features from the cleaned review
    feature_dict = get_feature_dict(cleaned_review)

    # Debugging: print the feature dictionary
    print(f"Feature dict: {feature_dict}")

    # Predict the category using the classifier
    prediction = classifier.classify(feature_dict)

    # Debugging: print the prediction
    print(f"Prediction: {prediction}")

    # Return the result as a JSON response
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
