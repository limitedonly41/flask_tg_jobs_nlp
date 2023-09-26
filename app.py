from flask import Flask, request, render_template, jsonify, url_for
# from utils import clean_text
import pickle
import time
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pymorphy2
import re
from sklearn.svm import LinearSVC
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from googletrans import Translator, constants
from pprint import pprint


translator = Translator()

app = Flask(__name__)


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("russian"))
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)



@app.route('/api/ml', methods=['GET', 'POST'])
def add_message():
	print('started')
	content = request.json
	t = str(content['text'])

	nltk.download('stopwords')
	# stop_words = stopwords.words('russian')
	# tokenizer = RegexpTokenizer(r'\w+')
	# text_1 = t.lower()
	# text_2 = text_1.replace("\\n", " ")
	# tokens = tokenizer.tokenize(text_2)
	# tokens_1 = [item for item in tokens if item not in stop_words]
	# text_3 = " ".join(["".join(txt) for txt in tokens_1])
	

	# def clean_text(text):
	# 	text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
	# 	text = text.lower()
	# 	text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
	# 	text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
	# 	text = " ".join(ma.parse(str(word))[0].normal_form for word in text.split())
	# 	text = ' '.join(word for word in text.split() if len(word) > 3)
	# 	#     text= text.encode("utf-8")

		# return text

	ma = pymorphy2.MorphAnalyzer()
	# text = clean_text(text_3)

	# print(text)
	# print('\n'*3)


	with open('vectorizer', 'rb') as vec:
		tfidf_vectorizer = pickle.load(vec)
	# with open('linear_svc', 'rb') as training_model:
	# 	model = pickle.load(training_model)
	with open('classifier_lr', 'rb') as training_model:
		lr_tfidf = pickle.load(training_model)
	with open('classes_cat.json') as json_file:
		classes_cat = json.load(json_file)


	translation = translator.translate(t.lower(), dest="ru")
	res = translation.text

	text_check = [res]


	X_test_preprocessed = [preprocess_text(text) for text in text_check]
	X_test_transformed = tfidf_vectorizer.transform(X_test_preprocessed)

	y_pred = lr_tfidf.predict(X_test_transformed)


	

	# y_proba = model.predict_proba(count_vect.transform([text]))
	# print(y_proba)
	# print('\n'*3)
	classes_cat_list = list(classes_cat.values())
	all_classes = {}
	for i, prob in enumerate(y_pred[0]):
		all_classes[classes_cat_list[i]] = prob



	return jsonify({"category": all_classes})

# TODO: add versioning to url
@app.route('/', methods=['GET', 'POST'])
def predict():


	# count_vect = CountVectorizer()
	""" Main webpage with user input through form and prediction displayed

	:return: main webpage host, displays prediction if user submitted in text field
	"""

	if request.method == 'POST':

		t = request.form['text']

		with open('vectorizer', 'rb') as vec:
			tfidf_vectorizer = pickle.load(vec)
		# with open('linear_svc', 'rb') as training_model:
		# 	model = pickle.load(training_model)
		with open('classifier_lr', 'rb') as training_model:
			lr_tfidf = pickle.load(training_model)
		with open('classes_cat.json') as json_file:
			classes_cat = json.load(json_file)

		translation = translator.translate(t.lower(), dest="ru")
		res = translation.text

		text_check = [t]
		X_test_preprocessed = [preprocess_text(text) for text in text_check]
		check_tfidf = tfidf_vectorizer.transform(X_test_preprocessed)

		y_pred = lr_tfidf.predict_proba(check_tfidf)

		classes_cat_list = list(classes_cat.values())
		all_classes = {}
		for i, prob in enumerate(y_pred[0]):
			all_classes[classes_cat_list[i]] = prob

		sortedDict = sorted(all_classes.items(), key=lambda x:x[1], reverse=True)

		cat_one = sortedDict[0][0]
	

		return render_template('index.html', answer=sortedDict, one_cat=cat_one)

	if request.method == 'GET':
		return render_template('index.html')

# # TODO: add versioning to api
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
