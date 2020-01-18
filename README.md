# Disaster Response Pipeline Project

### Instructions from Udacity that are relevant to setup:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


# Additional Notes and About
- This project is a part of my Udacity coursework.
- As a result of the data that's being used for the classifier (there is limited data on current features) the accuracy and results are lackluster.
- With more devotion and by ruling out some of the features, our accuracy could significantly improve.

# About
- This classifier is intended to provide the most important data feature ranked in a flask app.
- Follow the directions above in order to run the project locally!

# Relevant Python Repos needed for running locally
1. pandas
2. sqlalchemy
3. json
4. sys
5. plotly (graphing library)
6. nltk (stem, tokenize)
7. flask
8. sklearn (for models)
9. numpy
10. pickle
11. bz2 (for zipping up model data from pickle)
12. re (for some regex)
