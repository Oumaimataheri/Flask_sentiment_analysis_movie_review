from flask import Flask, render_template, url_for, request
import re
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
tokenizer = Tokenizer()

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
TRAIN_SIZE = 0.8
SEQUENCE_LENGTH = 300

app = Flask(__name__)
#Machine Learning code goes here
def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/moviereviews.tsv',sep='\t')
    # Features and Labels
    df.review = df.review.apply(lambda x: preprocess(x))
    df_x = df['review']
    df_y = df.label
    corpus = df_x
    df_train, df_test = train_test_split(df, test_size=1 - TRAIN_SIZE, random_state=42)
    documents = [_review.split() for _review in df_train.review]
    w2v_model = Word2Vec(documents, vector_size=200, window=7, min_count=1, workers=8, sg=1)
    tokenizer.fit_on_texts(df_train.review)
    vocab_size = len(tokenizer.word_index) + 1
    x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.review), maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.review), maxlen=SEQUENCE_LENGTH)
    encoder = LabelEncoder()
    encoder.fit(df_train.label.tolist())
    y_train = encoder.transform(df_train.label.tolist())
    y_test = encoder.transform(df_test.label.tolist())
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = pad_sequences(tokenizer.texts_to_sequences(data), maxlen=SEQUENCE_LENGTH)
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555, debug=True)

