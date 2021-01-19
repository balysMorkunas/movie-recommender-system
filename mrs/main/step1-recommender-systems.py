import numpy as np
import pandas as pd
import os.path
from random import randint
from tqdm import *

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
submission_filef1 = '../data/submissionf1.csv'
submission_filef2 = '../data/submissionf2.csv'
submission_filef3 = '../data/submissionf3.csv'



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

def predict_collaborative_filtering_useritem(movies, users, ratings, predictions):
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


    # 2. Compute the utility matrix containing the similarities between users

    # Pearson correlation
    # similarity_matrix = np.corrcoef(ratings_matrix)

    # Cosine for cosine and adj cosine similarities
    def cosine(a, b):
        p = a*b
        c, d = p / b, p / a
        c[np.isnan(c)] = 0
        d[np.isnan(d)] = 0
        norm = np.linalg.norm(c) * np.linalg.norm(d)
        if not norm:
            return 0
        return np.dot(a, b) / norm


    # # Cosine similarity
    def cosineMatrix(ratings):
        nousers = np.shape(ratings)[1]
        sim = np.zeros((nousers, nousers))
        for i in  tqdm(range(nousers)):
            for j in  range(i, nousers):
                sim[i][j] = cosine(ratings[:, i], ratings[:, j])
                sim[j][i] = sim[i][j]
        return sim
    similarity_matrix = cosineMatrix(ratings_matrix.T+1e-9)


    # Adjusted cosine similarity
    #
    # def adjustedCosineMatrix(ratings):
    #     nousers = np.shape(ratings)[0]
    #     noitems = np.shape(ratings)[1]
    #     sim = np.zeros((noitems, noitems))
    #     user_avg_rating = np.zeros(nousers)
    #     zero_items = 0
    #     for u in tqdm(range(nousers)):
    #         for i in range(noitems):
    #             if (ratings[u, i] == 0):
    #                 zero_items += 1
    #             user_avg_rating[u] += ratings[u, i]
    #         user_avg_rating[u] /= zero_items
    #     for i in tqdm(range(noitems)):
    #         for j in range(i, noitems):
    #             sim[i][j] = cosine(ratings[:, i] - user_avg_rating,
    #                                ratings[:, j] - user_avg_rating)
    #             sim[j][i] = sim[i][j]
    #     return sim
    #
    # similarity_matrix = adjustedCosineMatrix(ratings_matrix.T)

    # 3. Compute predictions

    # Function 1
    # def prediction_score (user, movie):
    #     sum_numer = 0
    #     # Add a small number to denominator to avoid division by 0
    #     sum_denomin = 1e-9
    #
    #     for idx, user_ratings in enumerate(ratings_matrix):
    #         # Skip over users that have not rated the movie in question
    #         if(user_ratings[movie] == 0):
    #             pass
    #         # Skip over the user we are predicting for, since he would not have rated the movie
    #         elif(idx == user):
    #             pass
    #         else:
    #             sum_numer += similarity_matrix[user, idx]*(user_ratings[movie])
    #             sum_denomin += similarity_matrix[user, idx]
    #
    #     return sum_numer/sum_denomin


    # # Get the average rating of each user
    avgs = ratings_description.groupby('userID').agg(['mean'])
    avgs.drop(['movieID'], axis=1, inplace=True)
    user_rating_avgs = avgs.to_numpy()
    print("user rating avgs", user_rating_avgs.shape)

    # Function 2
    # Predict the score of unknown rating. Function given in the link above
    # def prediction_score (user, movie):
    #     user_avg = user_rating_avgs[user]
    #     sum_numer = 0
    #     # Add a small number to denominator to avoid division by 0
    #     sum_denomin = 1e-9
    #
    #     for idx, user_ratings in enumerate(ratings_matrix):
    #         # Skip over users that have not rated the movie in question
    #         if(user_ratings[movie] == 0):
    #             pass
    #         # Skip over the user we are predicting for, since he would not have rated the movie
    #         elif(idx == user):
    #             pass
    #         else:
    #             sum_numer += similarity_matrix[user, idx]*(user_ratings[movie] -user_rating_avgs[idx])
    #             sum_denomin += similarity_matrix[user, idx]
    #
    #     return user_avg + sum_numer/sum_denomin


    # Function 3
    # # Another approach might be possible:
    # # Global baseline hybrid
    #
    #
    # Get the average rating of each movie
    # Copy the ratings matrix and calculate the mean over movies while not taking 0's in consideration.
    ratings_matrix_copy = ratings_matrix
    ratings_matrix_copy[ratings_matrix_copy == 0] = np.nan
    ratings_matrix_copy = ratings_matrix_copy.T
    movie_rating_avgs = np.nanmean(ratings_matrix_copy[:, 1:], axis=1)

    # Convert the nans back into 0's
    movie_rating_avgs[np.isnan(movie_rating_avgs)] = 0
    ratings_matrix_copy[np.isnan(ratings_matrix_copy)] = 0

    # Use user avgs for total avg
    total_avg = np.mean(user_rating_avgs)

    print("total avg", total_avg)
    print(movie_rating_avgs[1])
    print(user_rating_avgs[1]+ total_avg + movie_rating_avgs[1])


    def prediction_score (user, movie):
        user_avg = total_avg + user_rating_avgs[user] - total_avg + movie_rating_avgs[movie] - total_avg
        sum_numer = 0
        sum_denomin = 1e-9
        for idx, user_ratings in enumerate(ratings_matrix):
            if(user_ratings[movie] == 0):
                pass
            elif(idx == user):
                pass
            else:
                sum_numer += similarity_matrix[user, idx]*(user_ratings[movie] - (total_avg + user_rating_avgs[idx] - total_avg + movie_rating_avgs[movie] - total_avg))
                sum_denomin += similarity_matrix[user, idx]
        return user_avg + sum_numer/sum_denomin

    # Creating the final predictions format
    number_predictions = len(predictions)
    final_predictions = [[idx+1, prediction_score(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in tqdm(range(0, number_predictions ))]
    return final_predictions

# predictions = predict_collaborative_filtering_useritem(movies_description, users_description, ratings_description, predictions_description)


def predict_collaborative_filtering_itemitem(movies, users, ratings, predictions):
    # 0. Create the full ratings matrix - fill the missing movies with 0s

    # The ratings dataframe is missing some movies - meaning some movies were not rated by anyone.
    # We fix this by joining movies dataframe with ratings dataframe and filling the nan values with 0's.
    ratings = (movies_description.set_index("movieID")).join(
        ratings.pivot(index='movieID', columns='userID', values='rating'))
    ratings.drop(['year', 'movie'], axis=1, inplace=True)
    ratings = ratings.fillna(0)


    # 1. Create a user/movie matrix containing all of the ratings

    # Rows - users
    # Columns - movies
    ratings_matrix = ratings.to_numpy().T

    # 2. Compute the utility matrix containing the similarities between items

    # Pearson correlation
    similarity_matrix = np.corrcoef(ratings_matrix.T + 1e-9)

    # Cosine similarity
    # similarity_matrix = ratings_matrix.T.dot(ratings_matrix) + 1e-9
    # norms = np.array([np.sqrt(np.diagonal(similarity_matrix))])
    # similarity_matrix = (similarity_matrix / norms / norms.T)

    # # Adjusted Cosine similarity (my attempt)
    # avgs = ratings_description.groupby('userID').agg(['mean'])
    # avgs.drop(['movieID'], axis=1, inplace=True)
    # user_rating_avgs = avgs.to_numpy()
    #
    # ratings_transpose = ratings_matrix.T
    #
    # def adj_cosine_similarity(itemA, itemB):
    #
    #     users_that_rated_item1 = ratings_transpose[itemA].nonzero()
    #     users_that_rated_item2 = ratings_transpose[itemB].nonzero()
    #     users_that_rated_both_items = np.intersect1d(users_that_rated_item1, users_that_rated_item2)
    #
    #     sum_numer = 0
    #     # Add a small number to denominator to avoid division by 0
    #     sum_denomin = 1e-9
    #
    #     for user in users_that_rated_both_items:
    #
    #         sum_numer += (ratings_matrix[user, itemA]-user_rating_avgs[user])*(ratings_matrix[user, itemB]-user_rating_avgs[user])
    #         sum_denomin += np.sqrt((ratings_matrix[user, itemA]-user_rating_avgs[user])**2) * np.sqrt((ratings_matrix[user, itemB]-user_rating_avgs[user])**2)
    #
    #     return sum_numer/sum_denomin
    #
    #
    # noItems = len(ratings_transpose)
    #
    # adj_cosine_similarity_matrix = np.zeros((noItems, noItems))
    # for idx, itemA in tqdm(enumerate(ratings_transpose)):
    #     for idy, itemB in enumerate(ratings_transpose):
    #         adj_cosine_similarity_matrix[idx][idy] = adj_cosine_similarity(idx, idy)
    #         adj_cosine_similarity_matrix[idy][idx] = adj_cosine_similarity_matrix[idx][idy]

    # Adjusted cosine similarity (github)
    def cosine(a, b):
        p = a * b
        c, d = p / b, p / a
        c[np.isnan(c)] = 0
        d[np.isnan(d)] = 0
        norm = np.linalg.norm(c) * np.linalg.norm(d)
        if not norm:
            return 0
        return np.dot(a, b) / norm
    #
    # def cosineMatrix(ratings):
    #     noitems = np.shape(ratings)[1]
    #     sim = np.zeros((noitems, noitems))
    #     for i in  tqdm(range(noitems)):
    #         for j in  range(i, noitems):
    #             sim[i][j] = cosine(ratings[:, i], ratings[:, j])
    #             sim[j][i] = sim[i][j]
    #     return sim
    # similarity_matrix = cosineMatrix(ratings_matrix +1e-9)

    # def adjustedCosineMatrix(ratings):
    #     nousers = np.shape(ratings)[0]
    #     noitems = np.shape(ratings)[1]
    #     sim = np.zeros((noitems, noitems))
    #     user_avg_rating = np.zeros(nousers)
    #     zero_items = 0
    #     for u in tqdm(range(nousers)):
    #         for i in range(noitems):
    #             if (ratings[u, i] == 0):
    #                 zero_items += 1
    #             user_avg_rating[u] += ratings[u, i]
    #         user_avg_rating[u] /= zero_items
    #     for i in tqdm(range(noitems)):
    #         for j in range(i, noitems):
    #             sim[i][j] = cosine(ratings[:, i] - user_avg_rating,
    #                                ratings[:, j] - user_avg_rating)
    #             sim[j][i] = sim[i][j]
    #         # print i
    #     return sim
    #
    # similarity_matrix = adjustedCosineMatrix(ratings_matrix)

    # 3. Compute predictions
    # Function 1
    def prediction_scoref1 (user, movie):

        # Indexes of movies that the user has rated
        rated_movies = ratings_matrix[user].nonzero()[0]

        sum_numer = 0
        # Add a small number to denominator to avoid division by 0
        sum_denomin = 1e-9

        for rated_movie in rated_movies:

            # sum_numer += similarity_matrix[movie, rated_movie] * ratings_matrix[user, rated_movie]
            # sum_denomin += similarity_matrix[movie, rated_movie]

            # Adjusted cosine similarity
            sum_numer += similarity_matrix[movie, rated_movie] * ratings_matrix[user, rated_movie]
            sum_denomin += similarity_matrix[movie, rated_movie]

        return sum_numer/sum_denomin


    # # Get the average rating of each user
    avgs = ratings_description.groupby('userID').agg(['mean'])
    avgs.drop(['movieID'], axis=1, inplace=True)
    user_rating_avgs = avgs.to_numpy()
    print("user rating avgs", user_rating_avgs.shape)

    # Get the average rating of each movie
    # Copy the ratings matrix and calculate the mean over movies while not taking 0's in consideration.
    ratings_matrix_copy = ratings_matrix
    ratings_matrix_copy[ratings_matrix_copy == 0] = np.nan
    ratings_matrix_copy = ratings_matrix_copy.T
    movie_rating_avgs = np.nanmean(ratings_matrix_copy[:, 1:], axis=1)

    # Convert the nans back into 0's
    movie_rating_avgs[np.isnan(movie_rating_avgs)] = 0
    ratings_matrix_copy[np.isnan(ratings_matrix_copy)] = 0

    # Use user avgs for total avg
    total_avg = np.mean(user_rating_avgs)

    print("total avg", total_avg)
    print("moveie rating", movie_rating_avgs[2555])

    # Function 2
    def prediction_scoref2 (user, movie):
        movie_avg = movie_rating_avgs[movie]
        sum_numer = 0
        # Add a small number to denominator to avoid division by 0
        sum_denomin = 1e-9

        for idx, movie_ratings in enumerate(ratings_matrix.T):
            # Skip over movies that this user has not rated
            if(movie_ratings[user] == 0):
                pass
            # Skip over the movie we are predicting for
            elif(idx == movie):
                pass
            else:
                sum_numer += similarity_matrix[movie, idx]*(movie_ratings[user] -movie_rating_avgs[idx])
                sum_denomin += similarity_matrix[movie, idx]

        # print(movie_avg + sum_numer/sum_denomin)
        return movie_avg + sum_numer/sum_denomin

    # Function 3

    def prediction_scoref3(user, movie):
        movie_avg = total_avg + movie_rating_avgs[movie] - total_avg + user_rating_avgs[user] - total_avg
        sum_numer = 0
        sum_denomin = 1e-9
        for idx, movie_ratings in enumerate(ratings_matrix.T):
            if (movie_ratings[user] == 0):
                pass
            elif (idx == movie):
                pass
            else:
                sum_numer += similarity_matrix[movie, idx] * (movie_ratings[user] - (
                            total_avg + movie_rating_avgs[idx] - total_avg + user_rating_avgs[user] - total_avg))
                sum_denomin += similarity_matrix[movie, idx]
        return movie_avg + sum_numer / sum_denomin


    # Creating the final predictions format
    number_predictions = len(predictions)
    final_predictionsf1 = [[idx+1, prediction_scoref1(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in tqdm(range(0, number_predictions ))]
    final_predictionsf2 = [[idx+1, prediction_scoref2(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in tqdm(range(0, number_predictions ))]
    final_predictionsf3 = [[idx+1, prediction_scoref3(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in tqdm(range(0, number_predictions ))]

    return final_predictionsf1, final_predictionsf2, final_predictionsf3





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
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function


# Uncomment only when outputting submission


predictionsf1, predictionsf2, predictionsf3 = predict_collaborative_filtering_itemitem(movies_description, users_description, ratings_description, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_filef1, 'w') as submission_writer:
    # Formates data
    predictionsf1 = [map(str, row) for row in predictionsf1]
    predictionsf1 = [','.join(row) for row in predictionsf1]
    predictionsf1 = 'Id,Rating\n' + '\n'.join(predictionsf1)

    # Writes it dowmn
    submission_writer.write(predictionsf1)

with open(submission_filef2, 'w') as submission_writer:
    # Formates data
    predictionsf2 = [map(str, row) for row in predictionsf2]
    predictionsf2 = [','.join(row) for row in predictionsf2]
    predictionsf2 = 'Id,Rating\n' + '\n'.join(predictionsf2)

    # Writes it dowmn
    submission_writer.write(predictionsf2)

with open(submission_filef3, 'w') as submission_writer:
    # Formates data
    predictionsf3 = [map(str, row) for row in predictionsf3]
    predictionsf3 = [','.join(row) for row in predictionsf3]
    predictionsf3 = 'Id,Rating\n' + '\n'.join(predictionsf3)

    # Writes it dowmn
    submission_writer.write(predictionsf3)
