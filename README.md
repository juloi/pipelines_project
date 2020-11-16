# Disaster Response Pipeline Project
The goal of the project is to build a web app that allows a message in the context of a disaster to be give in into a search 
bar by a user, and predict multiple categories in which the message might fall into e.g. help request, direct report, weather related.

## Directories
The `notebook_preparation` directory is a set of notebooks used to interactively develop the classifier model.
The `final_pipeline_project` contains the final project directories.

## The data used
Under `final_pipeline_project/data` you will find the following:
* `disaster_messages.csv`: contains text messages that will be transformed into numeric value features using the Term Frequencies - 
Inverse Document Frequencies (TF-IDF) transformer.
* `disaster_categories.csv`: labeled categories that each of the messages belongs two. This is our target variable for classification.

## Classification
We are using a Random Forest classifier as the base model, extending for multiple output classification.

## How to execute the project:
1. Run the following commands in `final_pipeline_project` directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_messages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_messages.db models/classifier.pkl`

2. Run the following command in the `app` directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
