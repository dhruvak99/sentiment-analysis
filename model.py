import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import pickle
data = pd.read_csv('IMDB Dataset.csv',sep=',')
#print(data.head())
data['sentiment'] = data['sentiment'].map({'positive':1,'negative':0})
#print(data.head())

def length(text):
	count = 0
	for i in text.split():
		count+=1
	return count

data['length'] = data['review'].apply(lambda x:length(x))
#print(data.length.max())

def preprocessing(data):

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

	data['review'] = data.review.apply(lambda x:remove_URL(x))
	data['review'] = data.review.apply(lambda x:remove_tags(x))
	data['review'] = data.review.apply(lambda x:remove_emoji(x)) 
	data['review'] = data.review.apply(lambda x:remove_punct(x))
	data['review'] = data.review.apply(lambda x:remove_stopwords(x))


	return data['review']


data['review'] = preprocessing(data)

#print(data.head(10))
def counter_word(text):
	count = Counter()
	for i in text.values:
		for word in i.split():
			count[word]+=1
	return count

counter = counter_word(data.review)
data['length'] = data['review'].apply(lambda x:length(x))
#print(len(counter))
#print(counter.most_common(10))

#print(data.length.max())

#setting hyperparamters
NUM_WORDS = len(counter)
MAX_LENGTH = data.length.max()
OOV_TOKEN = "<OOV>"
EMBEDDING_DIM = 32
TRUNC_TYPE ="post"
EPOCHS = 20
BATCH_SIZE = 100
def train_test_split(data):

	train_size = int(data.shape[0]*0.8)

	X_train = data.review[0:train_size]
	y_train = data.sentiment[0:train_size]

	X_test = data.review[train_size:]
	y_test = data.sentiment[train_size:]

	return X_train,y_train,X_test,y_test


train_sentences,train_labels,test_sentences,test_labels = train_test_split(data)

#print(train_sentences[0:10])
#print(test_sentences[0:10])

def tokenizing_padding(train_sentences,test_sentences):

	tokenizer = Tokenizer(num_words = NUM_WORDS,oov_token = OOV_TOKEN)
	tokenizer.fit_on_texts(train_sentences)
	with open('tokenizer.pickle','wb') as handle:
		pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
	train_sequences = tokenizer.texts_to_sequences(train_sentences)
	train_padded = pad_sequences(sequences=train_sequences,padding='post',maxlen=MAX_LENGTH,truncating=TRUNC_TYPE)
	test_sequences = tokenizer.texts_to_sequences(test_sentences)
	test_padded = pad_sequences(sequences=test_sequences,maxlen=MAX_LENGTH)

	return train_padded,test_padded

train_padded,test_padded = tokenizing_padding(train_sentences,test_sentences)

#print(test_padded.shape)

def create_model(NUM_WORDS,EMBEDDING_DIM,MAX_LENGTH):
	model = tf.keras.Sequential([
                             tf.keras.layers.Embedding(NUM_WORDS,EMBEDDING_DIM,input_length=MAX_LENGTH),
                             tf.keras.layers.GlobalAveragePooling1D(),
                             tf.keras.layers.Dense(6,activation='relu'),
                             tf.keras.layers.Dense(1,activation='sigmoid')
    ])

	model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
	return model

model = create_model(NUM_WORDS,EMBEDDING_DIM,MAX_LENGTH)
print(model.summary())

def train_model(model,X_train,y_train,X_test,y_test,epochs,batch_size,callbacks):
	print('Training...')
	model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_test,y_test),callbacks=callbacks)
	return model.history

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
history = train_model(model,train_padded,train_labels,test_padded,test_labels,EPOCHS,BATCH_SIZE,[callback])
model.save('testmodel1.h5')

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()

plot_graphs(history,'accuracy')
plot_graphs(history,'loss')


prediction = model.predict_classes(test_padded)
print(classification_report(y_true=test_labels,y_pred=prediction))