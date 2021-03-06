# Fake news detection #

## Intro ##
This repository contains my bachelor thesis at the University. Two different methods are investigated to solve the problem of conveying inaccurate and untrustworthy information. In the first method, the structure and the content of the articles used are taken into account. 
Thus, Natural Language Processing techniques were applied to the articles and an LSTM Neural Network was trained to recognize whether an article is fake or not resulting in a credibility score. In contrast, the second method considers only the author’s credibility extracted from his behavior in social media using the deep learning model trained previously. Lastly, a hybrid model combines both methods to provide better results.

## Run ##

While on the root directory of the cloned repository
```
cd code && ./run.sh
```

## Architecture ##
The architecture of each model and the hybrid final are shown below:
![alt text](https://github.com/AndreasPapandreou/fake_news_detection/blob/master/res/ALGO1.png?raw=true)

![alt text](https://github.com/AndreasPapandreou/fake_news_detection/blob/master/res/ALGO2.png?raw=true)

![alt text](https://github.com/AndreasPapandreou/fake_news_detection/blob/master/res/FINAL_ALGO.png?raw=true)


## Some general info ##

This section describes the algorithm by following the below steps:

1. Run code/model_creation/callerHan1.py and extract a model and some weights for all headlines of our dataset.
2. Run code/model_creation/prepareModel and train our model given some articles from dataset.
3. Run code/model_creation/evaluateModel.py and evaluate the previous model.
4. Run code/model_creation/predictModel.py and predict some the credibilty of some new articles from our dataset.
5. Run code/score_authors/scoreAuthorsModelA.py and create some scores per author given his articles and the previous model.
6. Run code/score_authors/scoreAuthorsModelA_and_B.py and update the previous score per author using model B, which takes into account his behavior in twitter.
7. Run code/final_test/scoreArticlesModelA.py and predict the credibility per article given a new dataset taking into account only the model A.
8. Run code/final_test/scoreArticlesModelA_and_B.py and udpate the previous score per article taking into account the model B.

- - - -

The steps 1-4 use the data that are in database "postgres" and table "taggedarticles".  
The steps 5-6 use the data that are in database "authors" and tables "article", "author" and "commonFollowers".  
The steps 7-8 use the data that are in database "authors" and tables "article2" and "author".  

- - - -

The table "taggedarticles" from the database "postgres" contains articles that are used for the training and testing of our model.  
The table "article" from the database "authors" contains articles that are used for the scoring of their authors.  
The table "author" from the database "authors" contains some info about the authors that has been used.  
The table "commonFollowers" from the database "authors" contains some info about the number of common followers for each pair of authors.  
The table "article2" from the database "authors" contains articles that we need to predict their credibility using both of our final models.  

#### The dataset which is used in the last files (steps 7 and 8) is small. So, in order to get better and more credible results, more data  must be crawled!  

