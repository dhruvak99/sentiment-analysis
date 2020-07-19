from flask import Flask,render_template,url_for,request
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


app = Flask(__name__)

@app.route('/') #specify the URL it should trigger to execute the home function
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	model = load_model('testmodel1.h5')
	with open('tokenizer.pickle','rb') as handle:
		tokenizer = pickle.load(handle)
	if request.method == 'POST':
		message = request.form['message']
	def remove_URL(sample):
	    """Remove URLs from a sample string"""
	    return re.sub(r"http\S+", "", sample)

	TAG_RE = re.compile(r'<[^>]+>')
	def remove_tags(text):
		
	    return TAG_RE.sub('', text)

	def remove_emoji(string):
	    """
	    This method removes emojis,symbols and flags in the text
	    """
	    emoji_pattern = re.compile(
	      "["
	      u"\U0001F600-\U0001F64F" #emoticons
	      u"\U0001F300-\U0001F5FF" #symbols and pictographs
	      u"\U0001F680-\U0001F6FF" #transport and map symbols
	      u"\U0001F1E0-\U0001F1FF" #flags
	      u"\U00002702-\U000027B0"
	      u"\U000024C2-\U0001F251"
	      "]+",
	      flags=re.UNICODE
	    )
	    return emoji_pattern.sub("[^a-zA-Z]",string)

	def remove_punct(text):
	    """
	    This method removes all the punctions in the text
	    """
	    table = str.maketrans("","",string.punctuation)
	    return text.translate(table)

	stopwords = set(nltk.corpus.stopwords.words("english"))

	def remove_stopwords(text):
	    """
	    This method removes the stopwords from the text
	    """
	    text = [word.lower() for word in text.split() if word.lower() not in stopwords]
	    return " ".join(text)
	message = remove_stopwords(remove_punct(remove_emoji(remove_tags(remove_URL(message)))))
	data = [message]
	test_sequence = tokenizer.texts_to_sequences(data)
	test_padded = pad_sequences(test_sequence, maxlen=1429)
	test_prediction = model.predict_classes(test_padded)

	return render_template('result.html',prediction=test_prediction)

if __name__ == '__main__':
	app.run(debug=True)