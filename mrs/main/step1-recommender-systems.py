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

#####
##
## COLLABORATIVE FILTERING
##
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    # 0. Create the full ratings matrix - fill the missing movies with 0s

    # The ratings dataframe is missing some movies - meaning some movies were not rated by anyone.
    # We fix this by joining movies dataframe with ratings dataframe and filling the nan values with 0's.
    ratings = (movies_description.set_index("movieID")).join(
        ratings.pivot(index='movieID', columns='userID', values='rating'))
    ratings.drop(['year', 'movie'], axis=1, inplace=True)
    ratings = ratings.fillna(0)


    # 1. Create a user/movie matrix containing all of the ratings

    # Collaborative filtering can be done in 2 ways: user-item and item-item.
    # In this function we do user-item CF.
    ratings_matrix = ratings.to_numpy().T


    # 2. Compute the utility matrix containing the similarities between users (user_item) or items (item_item)

    # Value of user_item similarity_matrix[i][j] = similarity between user i and user j
    similarity_matrix = np.corrcoef(ratings_matrix)


    # 3. Compute predictions

    # "User-based nearest neighbor" section of this link:
    # https://medium.com/towards-artificial-intelligence/recommendation-system-in-depth-tutorial-with-python-for-netflix-using-collaborative-filtering-533ff8a0e444

    # Get the average rating of each user
    avgs = ratings_description.groupby('userID').agg(['mean'])
    avgs.drop(['movieID'], axis=1, inplace=True)
    user_rating_avgs = avgs.to_numpy()

    # Predict the score of unknown rating. Function given in the link above
    def prediction_score (user, movie):
        user_avg = user_rating_avgs[user]
        sum_numer = 0
        # Add a small number to denominator to avoid division by 0
        sum_denomin = 1e-9

        for idx, user_ratings in enumerate(ratings_matrix):
            # Skip over users that have not rated the movie in question
            if(user_ratings[movie] == 0):
                pass
            # Skip over the user we are predicting for, since he would not have rated the movie
            elif(idx == user):
                pass
            else:
                sum_numer += similarity_matrix[user, idx]*(user_ratings[movie] -user_rating_avgs[idx])
                sum_denomin += similarity_matrix[user, idx]

        return user_avg + sum_numer/sum_denomin

    # # Another approach might be possible:
    # # Global baseline hybrid
    #
    #
    # # Get the average rating of each movie
    # avgs = ratings.groupby('movieID').agg(['mean'])
    # avgs.drop(['userID'], axis=1, inplace=True)
    # movie_rating_avgs = avgs.to_numpy()
    #
    # total_avg = np.mean(movie_rating_avgs)
    #
    # def prediction_score (user, movie):
    #     user_avg = total_avg + user_rating_avgs[user] - total_avg + movie_rating_avgs[movie] - total_avg
    #     sum_numer = 0
    #     sum_denomin = 1e-9
    #     for idx, user_ratings in enumerate(ratings_matrix):
    #         if(user_ratings[movie] == 0):
    #             pass
    #         elif(idx == user):
    #             pass
    #         else:
    #             sum_numer += similarity_matrix[user, idx]*(user_ratings[movie] - (total_avg + user_rating_avgs[idx] - total_avg + movie_rating_avgs[movie] - total_avg))
    #             sum_denomin += similarity_matrix[user, idx]
    #
    #     return user_avg + sum_numer/sum_denomin

    # Creating the final predictions format
    number_predictions = len(predictions)
    final_predictions = [[idx+1, prediction_score(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in range(0, number_predictions )]
    return final_predictions

print(predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)[0])


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

# print(predict_rating([0.41, 0.59], [2, 3]))

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


# Uncomment only when outputting submission


# predictions = predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)
#
# # Save predictions, should be in the form 'list of tuples' or 'list of lists'
# with open(submission_file, 'w') as submission_writer:
#     # Formates data
#     predictions = [map(str, row) for row in predictions]
#     predictions = [','.join(row) for row in predictions]
#     predictions = 'Id,Rating\n' + '\n'.join(predictions)
#
#     # Writes it dowmn
#     submission_writer.write(predictions)
