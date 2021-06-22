# -*- coding: utf-8 -*-
# from __future__ import division

from keras import regularizers, constraints, optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Flatten
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras import backend as K

import numpy as np
from numpy import asarray
from numpy import array
from numpy import zeros

from sklearn.metrics import confusion_matrix
import psycopg2
import theano
import pickle
import sys
import re
import os

# -----------------------------------------------------------
# define some global configurations
# -----------------------------------------------------------
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['KERAS_BACKEND']='theano'

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
set_keras_backend("theano")
# -----------------------------------------------------------

class Preprocess:
	def __init__(self):
		self.EMBEDDING_DIM = int(os.environ['EMBEDDING_DIM']) 
		self.all_data = []
		self.all_headlines = []
		self.all_articles = []
		self.all_coverted_headlines = []
		self.all_coverted_articles = []
		self.all_scores = [] # 0 -> fake, 1 -> genuine
		self.metadata = {'articles':{}, 'headlines':{}}

	# this functions store all news to variables
	def readArticles(self):
		conn = psycopg2.connect(database=os.environ['POSTGRES_DB'], user = os.environ['POSTGRES_USER'], password = os.environ['POSTGRES_PASSWORD'], host = os.environ['POSTGRES_HOST'], port = os.environ['POSTGRES_PORT'])
		cur = conn.cursor()
		cur.execute("SELECT title_path, text_path, score from taggedarticles where tag='test'");
		rows = cur.fetchall()

		for i in range(0, len(rows)):
			headline = open(rows[i][0], 'r') 
			headline = headline.read()
			headline = headline.replace('\n', '')

			article = open(rows[i][1], 'r') 
			article = article.read()
			article = article.replace('\n', '')

			score = float(rows[i][2])

			if (len(headline) > 0 and len(article) > 0):		
				self.all_headlines.append(headline)
				self.all_articles.append(article)
				self.all_scores.append(score)

	# modify all articles
	def setArticles(self, data, aver_words, aver_sentences):		
		# create list with all articles removing some characters ('[', ']')
		all_articles = []

		tmp = data.split(',')
		each_article = []
		for j in range(0, len(tmp)):
			tmp[j] = tmp[j].replace('[', '')
			tmp[j] = tmp[j].replace(']', '')
			each_article.append(tmp[j])

		new_article = []
		if (len(each_article) < aver_sentences):
			for j in range(0,  len(each_article)):
				# keep only the first aver_words words of each article
				tmp = each_article[j].split(' ')

				if tmp[0] == '':
					tmp = tmp[1:]
				tmp = ' '.join(tmp[0:aver_words])
				new_article.append(tmp)
	
			
			for j in range(0,  aver_sentences - len(new_article)):
				new_article.append(' ')
		else:
			for j in range(0,  aver_sentences):
				# keep only the first aver_words words of each article
				tmp = each_article[j].split(' ')
				if tmp[0] == '':
					tmp = tmp[1:]
				tmp = ' '.join(tmp[0:aver_words])
				new_article.append(tmp)
		return new_article

	# prepare all data (articles and headlines) for tokenizer, create one list with all data
	def setData(self, articles, headlines):
		# store all articles and headlines
		all_data = []

		start = 0
		# for i in range(0, len(articles)):
		for j in range(0, len(articles)):
			all_data.append(articles[j])				
		all_data.append(headlines)
		
		# define end point
		end = start+j+1

		# store the indexeces for each article and its headline
		self.metadata['articles']['0'] = {}
		self.metadata['articles']['0']['start'] = start  
		self.metadata['articles']['0']['end'] = end			
		self.metadata['headlines']['0'] = {}
		self.metadata['headlines']['0']['index'] = end
		
		# define start point
		start = end+1
		self.all_data = all_data

	# this function converts categorical values to integers about all data and creates the embedding matrix	
	def convertDataAndCreateEmbeddingMatrix2(self, data, aver_words, final_tokenizer):
		# integer encode the documents
		encoded_docs = final_tokenizer.texts_to_sequences(data)

		# pad documents to a max length of 4 words
		max_length = aver_words
		padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
		self.all_data = padded_docs

	# create arrays for all articles and headlines setting the appropriate dimensions 
	def prepareData(self, articles, headlines):
		for i in range(0, len(headlines)):			
			if i > 0:
				final_headlines = np.insert(final_headlines, [1], [headlines[i]], axis=0)
			else:
				final_headlines = [headlines[i]] # 2 dimensions
			
		for i in range(0, len(articles)):			
			if (i > 0):
				final_articles = np.insert(final_articles, [1], [articles[i]], axis=0)
			else:				
				final_articles = [articles[i]] # 3 dimensions
		return final_articles, final_headlines

	def run(self):
		learning_rate = os.environ['LEARNING_RATE']
		aver_words = int(os.environ['AVER_WORDS'])
		aver_sentences = int(os.environ['AVER_SENTENCES']) 

		print ('Loading all articles...\n')
		self.readArticles()

		# loading
		with open('han1Results/tokenizer.pickle', 'rb') as handle:
		    final_tokenizer = pickle.load(handle)

		# Assuming your model includes instance of an "AttentionLayer" class
		att = AttentionLayer()

		print ('Loading our model...\n')
		loaded_model = load_model('test/final_model.h5', custom_objects={'AttentionLayer': att})
		sgd = optimizers.SGD(lr=learning_rate, decay=os.environ['DECAY'], momentum=os.environ['MOMENTUM'], nesterov=True)
		loaded_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

		predictions = []
		original = []
		print('Making predictions...\n')
		for i in range(0, len(self.all_articles)):
			each_article = self.setArticles(self.all_articles[i], aver_words, aver_sentences)

			self.setData(each_article, self.all_headlines[i])

			self.convertDataAndCreateEmbeddingMatrix2(self.all_data, aver_words, final_tokenizer)

			self.all_coverted_articles.append(self.all_data[self.metadata['articles']['0']['start']:self.metadata['articles']['0']['end']])
			self.all_coverted_headlines.append(self.all_data[self.metadata['headlines']['0']['index']])
			
			final_articles, final_headlines = self.prepareData(self.all_coverted_articles, self.all_coverted_headlines)
			final_articles = np.array(final_articles)
			final_headlines = np.array(final_headlines)

			current_pred = loaded_model.predict([final_articles, final_headlines])

			predictions.append(current_pred) #  verbose=0
			original.append(float(self.all_scores[i]))

			self.all_data = []
			self.all_coverted_articles = []
			self.all_coverted_headlines = []

		# round predictions
		rounded_pred = []
		for x in predictions:
			if (round(x[0]) >= float(0.5)):
				rounded_pred.append(0)
			else:
				rounded_pred.append(1)

		matrix = confusion_matrix(original, rounded_pred)
		print ('The confusion_matrix is {}.\n'.format(matrix))

class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(AttentionLayer,self).__init__(**kwargs)

	def build(self, input_shape):
		
		#print '\nhi in build attention'
		#print input_shape
	
		self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1], ), name='{}_W'.format(self.name), initializer = 'glorot_uniform', trainable=True)
		self.bw = self.add_weight(shape=(input_shape[-1], ), name='{}_b'.format(self.name), initializer = 'zero', trainable=True)
		self.uw = self.add_weight(shape=(input_shape[-1], ), name='{}_u'.format(self.name), initializer = 'glorot_uniform', trainable=True)
		self.trainable_weights = [self.W, self.bw, self.uw]
		
		#print "\nweights in attention"
		#print self.W._keras_shape
		#print self.bw._keras_shape
		#print self.uw._keras_shape
		super(AttentionLayer,self).build(input_shape)
	
	def compute_mask(self, input, mask):
        	return 2*[None]

	def call(self, x, mask=None):
	
		#print '\nhi in attention'
		#print x._keras_shape
		
		uit = K.dot(x, self.W)
		
		#print '\nuit'
		#print uit._keras_shape
		
		uit += self.bw
		uit = K.tanh(uit)

		ait = K.dot(uit, self.uw)
		a = K.exp(ait)

		# apply mask after the exp. will be re-normalized next
		#print mask
		if mask is not None:
			a *= K.cast(mask, K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		a = K.expand_dims(a)

		#print "in att ", K.shape(a)
			
		weighted_input = x * a
		
		#print weighted_input	
		
		ssi = K.sum(weighted_input, axis=1)
		#print "type ", type(ssi)	
		#print "in att si ", theano.tensor.shape(ssi)
		#1111print "hello"
		return [a, ssi]

	def get_output_shape_for(self, input_shape):
		return  [(input_shape[0],input_shape[1]), (input_shape[0], input_shape[-1])]

	def compute_output_shape(self, input_shape):
		#print input_shape
		return [(input_shape[0],input_shape[1]), (input_shape[0], input_shape[-1])]

if __name__ == '__main__':
	print('Process started!\n')
	dataset = Preprocess()
	dataset.run()
	print('Process finished!')