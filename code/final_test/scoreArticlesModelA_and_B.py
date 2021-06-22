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

class Process:
	def __init__(self):
		pass

	def run(self):
		conn = psycopg2.connect(database=os.environ['AUTHORS_DB'], user = os.environ['POSTGRES_USER'], password = os.environ['POSTGRES_PASSWORD'], host = os.environ['POSTGRES_HOST'], port = os.environ['POSTGRES_PORT'])		
		cur = conn.cursor()
		cur.execute("SELECT id, author, scoremodela from article2");
		rows = cur.fetchall()

		print('Iterating through all authors and score each one of them using both of models A and B...\n')
		for i in range(0, len(rows)):
			cur.execute("SELECT score from author where id="+str(rows[i][1])+";");
			rows2 = cur.fetchall()

			finalScore = float(0.5*float(rows[i][2])) + float(0.5*float(rows2[0][0]))

			# update article's score to table using model A and model B
			cur.execute("UPDATE ARTICLE2 SET SCOREMODELAMODELB = "+str(finalScore)+" WHERE id = '"+str(rows[i][0])+"';");
			conn.commit()

if __name__ == '__main__':
	print('Process started!\n')
	dataset = Process()
	dataset.run()
	print('Process finished!')