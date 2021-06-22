# -*- coding: utf-8 -*-
# from __future__ import division

import psycopg2
import sys
import os

# -----------------------------------------------------------
# define some global configurations
# -----------------------------------------------------------
reload(sys)
sys.setdefaultencoding('utf8')
# -----------------------------------------------------------

class Preprocess:
	def __init__(self):
		pass

	def getAllFollowers(self):
		self.followers = {}
		conn = psycopg2.connect(database=os.environ['AUTHORS_DB'], user = os.environ['POSTGRES_USER'], password = os.environ['POSTGRES_PASSWORD'], host = os.environ['POSTGRES_HOST'], port = os.environ['POSTGRES_PORT'])		
		cur = conn.cursor()

		# extract all authors
		cur.execute("SELECT id, username from author;");
		rows = cur.fetchall()
		authors = dict((x, y) for x, y in rows)

		# extract all commonFollowers
		cur.execute("SELECT author1, author2, followers from commonFollowers;");
		rows2 = cur.fetchall()
		for author1, author2, followers in rows2:
			if authors[author1] not in self.followers:
				self.followers[authors[author1]] = {authors[author2]:followers}
			else:
				self.followers[authors[author1]][authors[author2]] = followers

	def getFollowersPerAuthor(self, authorId):
		conn = psycopg2.connect(database=os.environ['AUTHORS_DB'], user = os.environ['POSTGRES_USER'], password = os.environ['POSTGRES_PASSWORD'], host = os.environ['POSTGRES_HOST'], port = os.environ['POSTGRES_PORT'])		
		cur = conn.cursor()

		# iterate through all authors
		cur.execute("SELECT username from author where id="+authorId+";");
		rows = cur.fetchall()
		return self.followers[str(rows[0][0])]

	def run(self):
		conn = psycopg2.connect(database=os.environ['AUTHORS_DB'], user = os.environ['POSTGRES_USER'], password = os.environ['POSTGRES_PASSWORD'], host = os.environ['POSTGRES_HOST'], port = os.environ['POSTGRES_PORT'])		
		cur = conn.cursor()

		# store each author's score
		scores = {}
		cur.execute("SELECT username, score from author");
		rows = cur.fetchall()
		for i in range(0, len(rows)):
			scores[str(rows[i][0])] = str(rows[i][1]) 

		# extract the common followers for the authors and store them to variable
		self.getAllFollowers()

		print('Iterating through all authors and update their score using model B...\n')
		# iterate through all authors
		cur.execute("SELECT id, username from author");
		rows2 = cur.fetchall()

		for i in range(0, len(rows2)):
			# compute author's score using modelB
			followersPerAuthor = self.getFollowersPerAuthor(str(rows2[i][0]))

			# compute the sum of author's followers
			sumOfFollowers = 0
			for key in followersPerAuthor:
				sumOfFollowers+=int(followersPerAuthor[key])

			# compute author's score using modelB
			scoreModelB = 0
			for key in followersPerAuthor:
				scoreModelB+=((float(followersPerAuthor[key])/sumOfFollowers) * float(scores[key])) 
			finalScore = float(0.5*float(scores[str(rows2[i][1])])) + float(0.5*scoreModelB) 
			cur.execute("UPDATE AUTHOR SET SCORE = "+str(finalScore)+" WHERE ID = "+str(rows2[i][0])+";");
			conn.commit()

if __name__ == '__main__':
	print('Process started!\n')
	dataset = Preprocess()
	dataset.run()
	print('Process finished!')