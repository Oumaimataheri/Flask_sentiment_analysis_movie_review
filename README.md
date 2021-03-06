# Flask_sentiment_analysis_movie_reviews:

-----------------

The project domains background around the area of Sentiment analysis or Opinion mining which is a significant task in the field of Natural Language Processing based on creating an automated process of analyzing text to determine the sentiment expressed (positive, negative ). It is also used to understand the sentiment in social media, in political analysis and in survey responses. In general the main aim of this is to determine the attitude of speaker with positive, neutral and negative polarity.

In this repository , i worked on a movie reviews dataset taken from : https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/blob/master/moviereviews.tsv

<p align="center">
  <img width="400" height="200" src="https://www.teamsupport.com/hubfs/6.29.20_week%20of%20Blog_Sentiment%20vs.%20CDI_Part%201_iStock-1134027636.jpg">
</p>





## **TASK** : 

The main task is to label phrases on a scale of two values: negative or positive. There are many obstacles such as sentence negation, sarcasm, language ambiguity, and many others make the sentiment prediction more difficult. In general, this particular Sentiment Analysis is a binary classification task to be faced. This Sentiment Classification Model is based on Word2Vec Vectors. 

## Understanding Word2Vec Model:
Word embeddings are words mapped to real number vectors such that it can capture the semantic meaning of words. The methods tried in my previous posts of BOW and TFIDF do not capture the meaning between the words, they consider the words seperately as features. Word embeddings use some models to map a word into vectors such that similar words will be closer to each other. As shown in the below figure, for example some of the positive words which are adjectives will be closer to each other and vice versa for negative adjectives. It captures semantical and syntactical information of words. To train this model it takes into consideration the words surrounding that word of particular window size. There are different ways of deriving the word embedding vectors. Word2vec is one such method where neural embeddings model is used to learn that. It uses following two architectures to achieve this.

- CBOW
- SKIP GRAM 

In this project , i'm using SKIP GRAM ;  it predicts embeddings for the surrounding context words in the specific window given a current word.
<p align="center">
  <img width="500" height="300" src="https://miro.medium.com/max/1959/1*MqoUdbWmPM8fQq8jzha-eg.png">
</p>


## Classification model : 
Once the Word2Vec vectors are ready for training , DecisionTreeClassifier is used here to do the sentiment classification in other words to predict the target labels. Decision tree classifier is Supervised Machine learning algorithm for classification.

## Deployment of  the ML using FLASK : 
You do not have to be a pro in HTML to build the front end of your application! First of all i create the frontend of my application then i Connect the webpage with the Model using FLASK .

## Result of the application: 

<img width="370" alt="screen1" src="https://user-images.githubusercontent.com/75047820/119992860-41509b00-bfcb-11eb-8183-3b0ec873edd0.png">

-----------------


<img width="307" alt="screen2" src="https://user-images.githubusercontent.com/75047820/119993143-8aa0ea80-bfcb-11eb-9339-8d5e013327f7.png">


## Requirements : 
There are some library requirements for the project and some which are specific to individual methods. The requirements are as follows:
- pandas
- nltk
- gensim
- sklearn
- keras
- flask


## Enjoy ! 

