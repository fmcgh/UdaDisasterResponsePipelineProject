# UdaDisasterResponsePipelineProject
Part of an academic course with Udacity.

# GitHub link
See main at: https://github.com/fmcgh/UdaDisasterResponsePipelineProject

# Project Overview
This project, part of a Udacity course, demonstrates how to build a machine learning pipeline for classifying disaster messages. It consists of three main components:

Data Cleaning (ETL Pipeline): Merge and clean raw disaster messages and category labels, storing the results in a SQLite database.

Model Building (ML Pipeline): Train a multi-label classifier to predict up to 36 disaster categories using NLP techniques.

Web Application: Create a simple Flask web app that lets users input messages and view classification results and interactive visualizations.

# Dataset Details
Two primary files from Appen are used:

disaster_messages.csv: Contains real messages from disaster events.

disaster_categories.csv: Provides corresponding category labels.

Key points include real-world examples, multi-label assignments, data cleaning to handle inconsistencies, and a diverse set of messages that challenge the model to perform well.

# Project Results
The final model accurately categorizes messages into relevant disaster types. The web app presents:

A bar chart for message categories,

A histogram of message lengths,

A chart of the top 10 words,

A heatmap showing category correlations.

Overall, the project illustrates a practical workflow from data cleaning to model training and deployment in a user-friendly interface.


# Setup and Installation
## Prerequisites
To run this project, you need Python 3.7 or higher and the following Python libraries:

Flask

pandas

numpy

plotly

scikit-learn

nltk

SQLAlchemy

joblib

wordcloud

# You can install these dependencies using pip:

pip install -r requirements.txt

# Database Setup
Download or generate the disaster_categories.csv and disaster_messages.csv files.
Use process_data.py to clean and insert the data into an SQLite database (InsertDatabaseName.db).
Model Training
Train the model by running the following command:

python models/train_classifier.py

The script will train a machine learning model using scikit-learn and save it as classifier.pkl in the models directory.

Running the Application
To start the Flask web application, run the run.py file:

python app/run.py
Once the application is running, open your browser and navigate to http://127.0.0.1:5000/ to access the web interface.

Features
Message Classification: Users can input disaster messages, and the system will classify the message into one or more predefined categories (e.g., weather, fire, flood, etc.).

# Data Visualizations:

Distribution of Message Categories: A bar chart showing the count of messages in each category.
Message Length Distribution: A histogram of the message lengths (in words).
Top 10 Most Frequent Words: A bar chart showing the most frequent words in the messages.
Category Correlations: A heatmap showing the correlation between different message categories.
Responsive Web Interface: The application is designed with a user-friendly web interface, where users can interact with the system and view the results.

# How it Works
Data Preprocessing: The process_data.py script processes the raw disaster message data (disaster_messages.csv) and converts it into a clean format before storing it in the SQLite database.

Model Training: The train_classifier.py script trains a machine learning classifier using scikit-learn. It uses a bag-of-words approach to convert text data into features and predicts which category or categories a message belongs to.

Flask Web App: The Flask app (run.py) hosts the web interface where users can enter a message for classification. The app also displays visualizations of the data using Plotly.

# Contributing
Feel free to fork the repository and use as you please. This was purely completed as part of an academic course with Udacity.

# License
This project is licensed under the MIT License - see the LICENSE file for details.




### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
