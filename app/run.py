import json
import plotly
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram, Heatmap
import joblib
from sqlalchemy import create_engine
from wordcloud import STOPWORDS


app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes and lemmatizes text to prepare it for model prediction.
    Excludes stopwords and punctuation marks.

    Args:
    text (str): The input text to be tokenized and lemmatized.

    Returns:
    list: A list of clean tokens after removing stopwords and punctuation.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(STOPWORDS)

    punctuation = set(string.punctuation)

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words and clean_tok not in punctuation:
            clean_tokens.append(clean_tok)

    return clean_tokens


def load_data():
    """
    Loads data from the SQLite database.

    Returns:
    pandas.DataFrame: A DataFrame containing the disaster response messages.
    """
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('disaster_messages', engine)
    return df


def load_model():
    """
    Loads the pre-trained model.

    Returns:
    object: The trained classification model.
    """
    model = joblib.load("../models/classifier.pkl")
    return model


@app.route('/')
@app.route('/index')
def index():
    """
    Displays the index webpage with visualizations of the disaster response data.
    
    The page includes visualizations like the distribution of message categories, 
    message lengths, the most frequent words, and correlations between categories.

    Returns:
    str: The rendered HTML page with embedded visualizations.
    """
    df = load_data()
    
    category_counts = df.drop(columns=['message', 'id', 'original', 'genre']).sum()
    category_names = category_counts.index.tolist()

    df['message_length'] = df['message'].apply(lambda x: len(x.split()))

    message_length_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]

    all_words = ' '.join(df['message'].tolist())
    tokens = tokenize(all_words)
    word_freq = pd.Series(tokens).value_counts().head(10)
    top_words = word_freq.index.tolist()
    top_word_counts = word_freq.values.tolist()

    category_corr = df.drop(columns=['message', 'id', 'original', 'genre']).corr()

    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        },
        {
            'data': [
                Histogram(
                    x=df['message_length'],
                    nbinsx=int(df['message_length'].max() / 5),
                    xbins=dict(start=0, end=100, size=5),
                    histnorm='count'
                )
            ],
            'layout': {
                'title': 'Distribution of Message Lengths',
                'xaxis': {'title': 'Message Length (Words)'},
                'yaxis': {'title': 'Count'},
                'bargap': 0.2
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_word_counts
                )
            ],
            'layout': {
                'title': 'Top 10 Most Frequent Words',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Word'}
            }
        },
        {
            'data': [
                Heatmap(
                    z=category_corr.values,
                    x=category_corr.columns,
                    y=category_corr.columns,
                    colorscale='Viridis'
                )
            ],
            'layout': {
                'title': 'Correlation Between Message Categories',
                'xaxis': {'title': 'Category'},
                'yaxis': {'title': 'Category'},
                'autosize': True,
                'xaxis': {'tickangle': 45}
            }
        }
    ]
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """
    Handles the user query for classifying a message.

    Takes the input from the user, uses the pre-trained model to classify the message,
    and displays the classification results.

    Returns:
    str: The rendered HTML page showing the classification results.
    """
    query = request.args.get('query', '') 
    model = load_model()
    
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Starts the Flask web application.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()