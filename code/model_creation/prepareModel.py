# -*- coding: utf-8 -*-
# from __future__ import division

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras import callbacks

import numpy as np
from numpy import asarray
from numpy import array
from numpy import zeros

from sklearn.model_selection import train_test_split, StratifiedKFold
from han3_pretrain import fhan3_pretrain
import psycopg2
import random
import pickle
import sys
import os

# -----------------------------------------------------------
# define some global configurations
# -----------------------------------------------------------
seed = 7
np.random.seed(seed)
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
		self.all_articles = []
		self.all_train_headlines = []
		self.all_train_articles = []
		self.all_coverted_headlines = []
		self.all_coverted_articles = []
		self.all_train_scores = [] # 0 -> fake, 1 -> genuine
		self.aver_words = []
		self.aver_sentences = []
		self.metadata = {'articles':{}, 'headlines':{}}

	# this functions store all news to variables
	def readArticles(self):
		conn = psycopg2.connect(database=os.environ['POSTGRES_DB'], user = os.environ['POSTGRES_USER'], password = os.environ['POSTGRES_PASSWORD'], host = os.environ['POSTGRES_HOST'], port = os.environ['POSTGRES_PORT'])
		# print "Opened database successfully"       
		cur = conn.cursor() 
		cur.execute("SELECT title_path, text_path, score, average_words, average_sentences, tag from taggedarticles where tag='train'");
		rows = cur.fetchall()

		for i in range(0, len(rows)):
			headline = open(rows[i][0], 'r') 
			headline = headline.read()
			headline = headline.replace('\n', '')

			article = open(rows[i][1], 'r') 
			article = article.read()
			article = article.replace('\n', '')

			score = float(rows[i][2])
			
			tag = str(rows[i][5])
			
			self.all_train_headlines.append(headline)
			self.all_train_articles.append(article)
			self.all_train_scores.append(score)

			self.aver_words.append(rows[i][3])
			self.aver_sentences.append(rows[i][4])


		# compute the average of words in all headlines and articles
		final_aver_words = self.getAverageWords()

		# compute the average of sentences in all headlines and articles
		final_aver_sentences = self.getAverageSentences()

		conn.close()
		return final_aver_words, final_aver_sentences

	def getAverageWords(self):
		summary = 0
		for i in range(0, len(self.aver_words)):
			summary += int(self.aver_words[i])
		final_aver_words = summary/len(self.aver_words)	
		return final_aver_words

	def getAverageSentences(self):
		summary = 0
		for i in range(0, len(self.aver_sentences)):
			summary += int(self.aver_sentences[i])
		final_aver_sentences = summary/len(self.aver_sentences)	
		return final_aver_sentences

	# modify all articles
	def setArticles(self, data, aver_words, aver_sentences):		
		# creatre list with all articles removing some characters ('[', ']')
		all_articles = []
		counter=0
		indeces = []

		for i in range(0, len(data)):
			tmp = data[i].split(',')
			each_article = []

			for j in range(0, len(tmp)):
				tmp[j] = tmp[j].replace('[', '')
				tmp[j] = tmp[j].replace(']', '')
				each_article.append(tmp[j])
				
			if len(each_article) > 5:
				counter+=1
				all_articles.append(each_article)
				indeces.append(i)

		# modify articles in order each one to have aver_sentences sentences
		for i in range(0, len(all_articles)):
			each_article = []
			if (len(all_articles[i]) < aver_sentences):
				for j in range(0,  len(all_articles[i])):
					# keep only the first aver_words words of each article
					tmp = all_articles[i][j].split(' ')
					if tmp[0] == '':
						tmp = tmp[1:]
					tmp = ' '.join(tmp[0:aver_words])
					each_article.append(tmp)

				for j in range(0,  aver_sentences - len(all_articles[i])):
					each_article.append(' ')
			else:
				for j in range(0,  aver_sentences):
					# keep only the first aver_words words of each article
					tmp = all_articles[i][j].split(' ')
					if tmp[0] == '':
						tmp = tmp[1:]
					tmp = ' '.join(tmp[0:aver_words])
					each_article.append(tmp)
			
			self.all_articles.append(each_article)
		return indeces

	# prepare all data (articles and headlines) for tokenizer, create one list with all data
	def setData(self, articles, headlines):
		# store all articles and headlines
		all_data = []

		start = 0
		for i in range(0, len(articles)):
			for j in range(0, len(articles[i])):
				all_data.append(articles[i][j])				
			all_data.append(headlines[i])
			
			# define end point
			end = start+j+1

			# store the indexeces for each article and its headline
			self.metadata['articles'][str(i)] = {}
			self.metadata['articles'][str(i)]['start'] = start  
			self.metadata['articles'][str(i)]['end'] = end			
			self.metadata['headlines'][str(i)] = {}
			self.metadata['headlines'][str(i)]['index'] = end
			
			# define start point
			start = end+1
		self.all_data = all_data

	# this function converts categorical values to integers about all data and creates the embedding matrix	
	def convertDataAndCreateEmbeddingMatrix(self, data, aver_words, t):
		vocab_size = len(t.word_index) + 1

		# integer encode the documents
		encoded_docs = t.texts_to_sequences(data)

		# pad documents to a max length of max_length words
		max_length = aver_words
		padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
		self.all_data = padded_docs

		# print ("Creating word embedding matrix")
		#TODO: get word2vec pre trained on news corpus
		GLOVE_DIR = os.environ['GLOVE_DIR']
		embeddings_index = {}
		f = open(os.path.join(GLOVE_DIR, os.environ['GLOVE_OPTION']))
		for line in f:
			values = line.split()
			#glove file contains a word and then list of values which are the coefficients corresponding to the k-dimensional embedding
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
		print('Total words in embedding dictionary: {}\n'.format(len(embeddings_index)))

		#creating final matrix just for vocabulary words
		#all elements in particular the zeroth element is initialized to all zeroes 
		#all elements except the zeroth element will be changed later
		# initialize embedding_matrix
		embedding_matrix = np.zeros(shape=(vocab_size, self.EMBEDDING_DIM), dtype='float32')
		
		embedding_matrix[1] = np.random.uniform(-0.25, 0.25, self.EMBEDDING_DIM)
		for word, i in t.word_index.items():
		    embedding_vector = embeddings_index.get(word) #glove coeffs wrt to the words
		    if embedding_vector is not None:
		        # words not found in embedding index will be all-zeros.
		        embedding_matrix[i] = embedding_vector
		    #0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
		    #ref: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py                              
		    else:
		        embedding_matrix[i] = np.random.uniform(-0.25,0.25, self.EMBEDDING_DIM) 

		print('Total words in embedding matrix: {}\n'.format(len(embedding_matrix)))
		# return embedding_matrix, vocab_size, t
		return embedding_matrix, vocab_size

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
		print ('Loading all articles...\n')
		average_words, average_sentences = self.readArticles()

		print ('Setting all articles...\n')
		indeces = self.setArticles(self.all_train_articles, average_words, average_sentences)

		# in set Articles we use only article with length > aver_sentences/2, so we need to update the self.all_train_headlines
		new_all_train_headlines = []
		new_all_train_scores = []
		for i in range(0, len(self.all_train_headlines)):
			if i in indeces:
				new_all_train_headlines.append(self.all_train_headlines[i])
				new_all_train_scores.append(self.all_train_scores[i])
		self.all_train_headlines = new_all_train_headlines
		self.all_train_scores = new_all_train_scores

		print ('Setting all data (articles and headlines)...\n')
		self.setData(self.all_articles, self.all_train_headlines)

		# loading
		with open('han1Results/tokenizer.pickle', 'rb') as handle:
		    final_tokenizer = pickle.load(handle)

		print('Creating embedding matrix...\n')
		embedding_matrix, vocab_size = self.convertDataAndCreateEmbeddingMatrix(self.all_data, average_words, final_tokenizer)

		print('Preparing our final data...\n')
		for i in range(0, len(self.all_train_headlines)):
			self.all_coverted_articles.append(self.all_data[self.metadata['articles'][str(i)]['start']:self.metadata['articles'][str(i)]['end']])
			self.all_coverted_headlines.append(self.all_data[self.metadata['headlines'][str(i)]['index']])
		final_train_articles, final_train_headlines = self.prepareData(self.all_coverted_articles, self.all_coverted_headlines)

		# split dataset to train and validation set
		final_data = {}
		final_data['train_articles'] = []
		final_data['train_headlines'] = []
		final_data['train_scores'] = []

		# convert list to numpy array
		final_data['train_articles'] =  final_train_articles
		final_data['train_headlines'] = final_train_headlines
		final_data['train_scores'] = np.array(self.all_train_scores)

		final_data['embedding_matrix'] = embedding_matrix
		final_data['average_words'] = average_words
		final_data['average_sentences'] = average_sentences
		final_data['vocab_size'] = vocab_size
		return final_data

class Model:
	def __init__(self):
		self.EMBEDDING_DIM = int(os.environ['EMBEDDING_DIM'])  

	def run(self, results):
		WORD_LIMIT = int(1*results['average_words'])
		SENT_LIMIT = int(1*results['average_sentences'])
		WORDGRU = self.EMBEDDING_DIM/2
		dropoutper = os.environ['DROPOUTPER']
		learning_rate = os.environ['LEARNING_RATE']

		print ('Building our model using StratifiedKFold with 5-folds...\n')

		model, model1 = fhan3_pretrain(results['vocab_size'], WORD_LIMIT, SENT_LIMIT, self.EMBEDDING_DIM, WORDGRU, results['embedding_matrix'], dropoutper, learning_rate)
		# print (model.summary())
		# print (model1.summary())
		# print (model.get_config())
		
		class ModelSave(callbacks.Callback):
		    def on_epoch_end(self, epoch, logs={}):
		        model1.save('test/SavedModels/han3_pretrain_wordEncoder_epoch_{}.h5'.format(epoch))
		        model.save('test/SavedModels/han3_pretrain_epoch_{}.h5'.format(epoch))
		        
		modelsave = ModelSave()
		new_callbacks = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto'), modelsave]

		# -----------------------------------------------------------
		# Create five models using StratifiedKFold in order to avoid 
		# overffiting using k=5. If the results are satisfactory
		# we create the final model given all of our data
		# -----------------------------------------------------------

		# -----------------------------------------------------------
		# create 5 models in order to prevent overffiting
		# -----------------------------------------------------------
		
		# k = 5
		# folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=7).split(results['train_articles'], results['train_scores']))
		# for m, (train_idx, val_idx) in enumerate(folds):
		# 	print('Fold {} has started...\n'.format(m))
		# 	X = results['train_articles'][train_idx]
		# 	Y = results['train_headlines'][train_idx]
		# 	Z = results['train_scores'][train_idx]

		# 	K = results['train_articles'][val_idx]
		# 	L = results['train_headlines'][val_idx]
		# 	M = results['train_scores'][val_idx]

		# 	history = model.fit([X, Y], Z, validation_data=([K , L], M), shuffle=True, batch_size=int(os.environ['BATCH_SIZE']), epochs=int(os.environ['FHAN3_PRETRAIN_EPOCHS']), callbacks=new_callbacks)
		# 	print('Saving our model...\n')
		# 	model.save('test/model_fold_'+str(m)+'.h5')  # creates a HDF5 file 'my_model.h5'
		# 	train_eval = model.evaluate([X, Y], Z, batch_size=int(os.environ['BATCH_SIZE']))
		# 	valid_eval = model.evaluate([K, L], M, batch_size=int(os.environ['BATCH_SIZE']))

		# ----------------------
		# create our final model
		# ----------------------
		history = model.fit([results['train_articles'], results['train_headlines']], results['train_scores'], shuffle=True, batch_size=int(os.environ['BATCH_SIZE']), epochs=int(os.environ['FHAN3_PRETRAIN_EPOCHS']), callbacks=new_callbacks)
		print('Saving our model...\n')
		model.save('test/final_model.h5')  # creates a HDF5 file 'my_model.h5'
		return model

if __name__ == '__main__':
	print('Process started!\n')
	dataset = Preprocess()
	results = dataset.run()

	myModel = Model()
	model = myModel.run(results)
	print('Process finished!')