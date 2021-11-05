from flask import Flask, request, render_template, jsonify, url_for
# from utils import clean_text
import pickle
import time
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC


app = Flask(__name__)



# TODO: add versioning to url
@app.route('/', methods=['GET', 'POST'])
def predict():
    # count_vect = CountVectorizer()
    """ Main webpage with user input through form and prediction displayed

    :return: main webpage host, displays prediction if user submitted in text field
    """

    if request.method == 'POST':

        text = request.form['text']
        # input_text = clean_text(response)
        # input_text = vectorizer.transform([input_text])
        # prediction = model.predict(input_text)
        # prediction = 'Cyber-Troll' if prediction[0] == 1 else 'Non Cyber-Troll'
        with open('vectorizer', 'rb') as vec:
            count_vect = pickle.load(vec)
        with open('multiclass_classifier_rf', 'rb') as training_model:
            model = pickle.load(training_model)
        print(model.predict(count_vect.transform(['Ищу таргетолога. Рассматриваю начинающих специалистов тоже. Писать в лс'])))
        response = model.predict(count_vect.transform([str(text)]))

        return render_template('index.html', text=text, submission=str(response[0]))

    if request.method == 'GET':
        return render_template('index.html')

# TODO: add versioning to api
# @app.route('/predict', methods=['POST'])
# def predict_api():
#     """ endpoint for model queries (non gui)

#     :return: json, model prediction and response time
#     """
#     start_time = time.time()

#     # request_data = request.json
#     # input_text = request_data['data']
#     # input_text = clean_text(input_text)
#     # input_text = vectorizer.transform([input_text])
#     # prediction = model.predict(input_text)
#     # prediction = 'Cyber-Troll' if prediction[0] == 1 else "Non Cyber-Troll"  # post processing

#     response = {'prediction': 'prediction', 'response_time': time.time() - start_time}
#     print(response)
#     return jsonify(response)


if __name__ == '__main__':
    app.run(debug=False)
