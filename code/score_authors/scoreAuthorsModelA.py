# -*- codifor each pair of usersng: utf-8 -*-
# from __future__ import division

from keras import regularizers, constraints, optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras.models import load_model
from keras import backend as K
from keras.layers import Input
from keras import callbacks

import numpy as np
from numpy import asarray
from numpy import array
from numpy import zeros

import psycopg2
import theano
import pickle
import sys
import re
import os

# -----------------------------------------------------------
# define some global configurations
# -----------------------------------------------------------
os.environ['KERAS_BACKEND']='theano'
reload(sys)
sys.setdefaultencoding('utf8')

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
		self.metadata = {'articles':{}, 'headlines':{}}

	# this functions store all news to variables
	def readArticles(self, author):
		conn = psycopg2.connect(database=os.environ['AUTHORS_DB'], user = os.environ['POSTGRES_USER'], password = os.environ['POSTGRES_PASSWORD'], host = os.environ['POSTGRES_HOST'], port = os.environ['POSTGRES_PORT'])		
		cur = conn.cursor()
		cur.execute("SELECT title_path, text_path from article where author="+str(author)+"");
		rows = cur.fetchall()

		for j in range(0, len(rows)):
			headline = open(rows[j][0], 'r') 
			headline = headline.read()
			headline = headline.replace('\n', '')

			article = open(rows[j][1], 'r') 
			article = article.read()
			article = article.replace('\n', '')

			if (len(headline) > 0 and len(article) > 0):		
				self.all_headlines.append(headline)
				self.all_articles.append(article)

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
	def convertDataAndCreateEmbeddingMatrix(self, data, aver_words, final_tokenizer):
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
		aver_words = int(os.environ['AVER_WORDS'])
		aver_sentences = int(os.environ['AVER_SENTENCES']) 

		# loading
		print('Getting the tokenizer from ../model_creation/han1Results/ \n')
		with open('../model_creation/han1Results/tokenizer.pickle', 'rb') as handle:
		    final_tokenizer = pickle.load(handle)

		# Assuming your model includes instance of an "AttentionLayer" class
		att = AttentionLayer()
		
		print('Loading our model...\n')
		loaded_model = load_model('../model_creation/test/final_model.h5', custom_objects={'AttentionLayer': att})
		sgd = optimizers.SGD(lr=os.environ['LEARNING_RATE'], decay=os.environ['DECAY'], momentum=os.environ['MOMENTUM'], nesterov=True)
		loaded_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

		conn = psycopg2.connect(database=os.environ['AUTHORS_DB'], user = os.environ['POSTGRES_USER'], password = os.environ['POSTGRES_PASSWORD'], host = os.environ['POSTGRES_HOST'], port = os.environ['POSTGRES_PORT'])		
		cur = conn.cursor()
		# iterate through all authors
		cur.execute("SELECT id from author");
		rows2 = cur.fetchall()

		print('Iterating through all authors and score each one of them using model A...\n')
		# compute author's score using modelA
		for i in range(0, len(rows2)):
			self.all_headlines = []
			self.all_articles = []
			self.readArticles(str(rows2[i][0]))		

			predictions = []
			metadata = {}
			for m in range(0, len(self.all_articles)):

				each_article = self.setArticles(self.all_articles[m], aver_words, aver_sentences)

				self.setData(each_article, self.all_headlines[m])
				self.convertDataAndCreateEmbeddingMatrix(self.all_data, aver_words, final_tokenizer)
				self.all_coverted_articles.append(self.all_data[self.metadata['articles']['0']['start']:self.metadata['articles']['0']['end']])
				self.all_coverted_headlines.append(self.all_data[self.metadata['headlines']['0']['index']])

				final_articles, final_headlines = self.prepareData(self.all_coverted_articles, self.all_coverted_headlines)

				final_articles = np.array(final_articles)
				final_headlines = np.array(final_headlines)

				current_pred = loaded_model.predict([final_articles, final_headlines])

				predictions.append(current_pred)

				self.all_data = []
				self.all_coverted_articles = []
				self.all_coverted_headlines = []

			scoreModelA = sum(predictions)/len(predictions)

			cur.execute("UPDATE AUTHOR SET SCORE = "+str(scoreModelA[0][0])+" WHERE ID = "+str(rows2[i][0])+";");
			conn.commit()

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