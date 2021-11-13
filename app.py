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

app = Flask(__name__)

@app.route('/api/ml', methods=['GET', 'POST'])
def add_message():
	content = request.json
	t = str(content['text'])

	nltk.download('stopwords')
	stop_words = stopwords.words('russian')
	tokenizer = RegexpTokenizer(r'\w+')
	text_1 = t.lower()
	text_2 = text_1.replace("\\n", " ")
	tokens = tokenizer.tokenize(text_2)
	tokens_1 = [item for item in tokens if item not in stop_words]
	text_3 = " ".join(["".join(txt) for txt in tokens_1])
	print(text_3)
	def clean_text(text):
		text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
		text = text.lower()
		text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
		text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
		text = " ".join(ma.parse(str(word))[0].normal_form for word in text.split())
		text = ' '.join(word for word in text.split() if len(word) > 3)
		#     text= text.encode("utf-8")

		return text

	ma = pymorphy2.MorphAnalyzer()
	text = clean_text(text_3)

	with open('vectorizer', 'rb') as vec:
		count_vect = pickle.load(vec)
	with open('multiclass_classifier_rf', 'rb') as training_model:
		model = pickle.load(training_model)
	with open('classes_cat.json') as json_file:
		classes_cat = json.load(json_file)

	y_proba = model.predict_proba(count_vect.transform([text]))
	all_classes = []
	for prob in y_proba:
		res = [0]
		tags = [classes_cat[str(list(prob).index(max(prob)))]]
		s = sorted(list(prob), reverse=True)
		for v in s[1:]:
			if max(prob)/v < 2:
				tags.append(classes_cat[str(list(prob).index(v))])
		all_classes.append(tags)
	return jsonify({"category": all_classes})

# # TODO: add versioning to url
# @app.route('/', methods=['GET', 'POST'])
# def predict():
#     # count_vect = CountVectorizer()
#     """ Main webpage with user input through form and prediction displayed

#     :return: main webpage host, displays prediction if user submitted in text field
#     """

#     if request.method == 'POST':
# 		# df = pd.read_csv(request.files.get('file'))
# 		# input_text = clean_text(response)
# 		# input_text = vectorizer.transform([input_text])
# 		# prediction = model.predict(input_text)
# 		# prediction = 'Cyber-Troll' if prediction[0] == 1 else 'Non Cyber-Troll'
# 		with open('vectorizer', 'rb') as vec:
# 		    count_vect = pickle.load(vec)
# 		with open('multiclass_classifier_rf', 'rb') as training_model:
# 		    model = pickle.load(training_model)

# 		svm = CalibratedClassifierCV(base_estimator=model, cv='prefit')
# 		svm.fit(X_train_tfidf, y_train)
# 		y_proba = svm.predict_proba(count_vect.transform(df))
# 		# df['class'] = y_proba
# 		# resp = make_response(df.to_csv())
# 		# resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
# 		# resp.headers["Content-Type"] = "text/csv"
# 		return resp

    # if request.method == 'GET':
    #     return render_template('index.html')

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
