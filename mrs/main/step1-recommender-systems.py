import numpy as np
import pandas as pd
import os.path
from random import randint

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = '../data/movies.csv'
users_file = '../data/users.csv'
ratings_file = '../data/ratings.csv'
predictions_file = '../data/predictions.csv'
submission_file = '../data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

# The ratings dataframe is missing some movies - meaning some movies were not rated by anyone.
# We fix this by joining movies dataframe with ratings dataframe and filling the nan values with 0's.
ratings_full = (movies_description.set_index("movieID")).join(ratings_description.pivot(index='movieID', columns='userID', values='rating'))

ratings_full.drop(['year', 'movie'], axis=1, inplace=True)
ratings_full = ratings_full.fillna(0)

#####
##
## COLLABORATIVE FILTERING
##
## In hindishgt we want movie-movie collab. filtering.
## For movie x find two other most-similar movies .
## Predict rating x based on neighbours.
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    # TO COMPLETE

    pass


def central_cosine_distance(vecA, vecB):
    """
    Calculates Pearsons Correlation between two vectors.
    :param vecA: list
    :param vecB: list
    :return: float
    """
    meanA = np.mean(vecA)
    meanB = np.mean(vecB)
    sum_numerator = np.sum((vecA - meanA) * (vecB - meanB))
    sum_denominator = np.sqrt(np.sum((vecA - meanA) ** 2)) * np.sqrt(np.sum((vecB - meanB) ** 2))

    return sum_numerator/sum_denominator

print(central_cosine_distance([-2.6, 0, -0.6, 0, 0, 1.4, 0, 0, 1.4, 0, 0.4, 0],
                              [-2.6, 0, -0.6, 0, -0.6, 0, 0, -1.6, 0, 0, 0.4, 0]))


def predict_rating(s, r):
    """
    Predict rating by taking weighted average of neighbour ratings.
    :param s: list
        Closest neighbour similarities.
    :param r:
        Closes neighbour actual ratings for item we want to predict.
    :return:
    """

    sumNumerator = np.sum(np.dot(s, r).ravel())
    sumDenominator = np.sum(s)

    return sumNumerator/sumDenominator

print(predict_rating([0.41, 0.59], [2, 3]))

#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####

# By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)
