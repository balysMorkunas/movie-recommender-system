import numpy as np
import pandas as pd
import os.path
from random import randint
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
# DATA IMPORT
##
#####


# Where data is located
movies_file = '../data/movies.csv'
users_file = '../data/users.csv'
ratings_file = '../data/ratings.csv'
predictions_file = '../data/predictions.csv'
submission_file = '../data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';',
                                 dtype={'movieID': 'int',
                                        'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int',
                                       'gender': 'str',
                                       'age': 'int',
                                       'profession': 'int'},
                                names=['userID', 'gender',
                                       'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int',
                                         'movieID': 'int',
                                         'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';',
                                      names=['userID', 'movieID'], header=None)

# The ratings dataframe is missing some movies -
# meaning some movies were not rated by anyone.
# We fix this by joining movies dataframe with
# ratings dataframe and filling the nan values with 0's.
ratings_test = pd.DataFrame(np.array([
                [0, 0, 1, 2, 3, 0, 0, 1, 0, 2, 4, 1, 5, 2],
                [0, 0, 0, 0, 0, 4, 2, 1, 4, 0, 0, 0, 2, 2],
                [1, 0, 0, 2, 4, 0, 0, 0, 0, 1, 4, 0, 0, 2],
                [0, 1, 4, 0, 0, 0, 2, 1, 2, 0, 0, 0, 1, 1],
                [4, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 1, 2, 0],
                [0, 5, 1, 0, 0, 1, 2, 1, 1, 1, 0, 0, 0, 2]]))
#####
##
# COLLABORATIVE FILTERING
##
# In hindsight we want movie-movie collab. filtering.
# For movie x find two other most-similar movies .
# Predict rating x based on neighbours.
#####


def predict_collaborative_filtering(movies, users, ratings, predictions):
    # 1. Create a user/movie matrix containing all of the ratings

    # Collaborative filtering can be done in 2 ways: user-item and item-item.
    # An easy way to switch between User-Item CF and Item-Item CF:
    user_item = True

    # If user_item is true, we have a matrix where rows are users and columns
    # are movies. The first row of this matrix has ratings
    # of user 1 for movies 1-3695
    # If user_item is false, then we have a matrix where rows are movies and
    # columns are users. The first row of this matrix has ratings of
    # movie 1 for users 6040

    if(user_item):
        ratings_matrix = ratings.to_numpy().T
    else:
        ratings_matrix = ratings.to_numpy()

    # 2. Compute the utility matrix containing the similarities
    # between users (user_item) or items (item_item)

    # Value of user_item
    # similarity_matrix[i][j] = similarity between user i and user j
    # similarity_matrix = np.corrcoef(ratings_matrix)

    if (user_item):
        similarity_matrix = ratings_matrix.dot(ratings_matrix.T) + 1e-9
    else:
        similarity_matrix = ratings_matrix.T.dot(ratings_matrix) + 1e-9
    norms = np.array([np.sqrt(np.diagonal(similarity_matrix))])
    similarity_matrix = similarity_matrix / norms / norms.T
    print(similarity_matrix)

    # 3. Compute predictions
    if(user_item):
        prediction_matrix = similarity_matrix.dot(
            ratings_matrix)/np.array([np.abs(similarity_matrix).sum(axis=1)]).T
    else:
        prediction_matrix = (ratings_matrix.dot(
            similarity_matrix)/np.array(
                [np.abs(similarity_matrix).sum(axis=1)])).T
    # print(predictidon_matrix)
    # Creating the final predictions format
    number_predictions = len(predictions)
    final_predictions = [[idx+1, prediction_matrix[predictions.userID[idx]-1,
                                                   predictions.movieID[idx]-1]]
                         for idx in range(0, number_predictions)]

    return final_predictions


#print(predict_collaborative_filtering(
#    movies_description,
#    users_description,
#    ratings_full,
#    predictions_description)[0])


def central_cosine_distance(vecA, vecB):
    """
    Calculates Pearsons Correlation between two vectors.
    :param vecA: list
    :param vecB: list
    :return: float
    """
    # meanA = np.mean(vecA)
    # meanB = np.mean(vecB)
    # sum_numerator = np.sum((vecA - meanA) * (vecB - meanB))
    # sum_denominator = np.sqrt(np.sum((vecA - meanA) ** 2)) * np.sqrt(np.sum((vecB - meanB) ** 2))
    #
    # return sum_numerator/sum_denominator

    # Visa sita padaro np.corrcoef funkcija
    return np.corrcoef(vecA, vecB)


# print(central_cosine_distance([-2.6, 0, -0.6, 0, -0.6, 0, 0, -1.6, 0, 0, 0.4, 0],
#                               [-2.6, 0, -0.6, 0, -0.6, 0, 0, -1.6, 0, 0, 0.4, 1]))

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
    # Baseline estimate should be added here!!! (baselineEst = overall_mean_rating
    #                                                          + rating_deviation_of_user
    #                                                          + (avg. rating of movie - mean))
    return sumNumerator/sumDenominator

#print(predict_rating([0.41, 0.59], [2, 3]))

#####
##
# LATENT FACTORS
##
#####


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


def train_test_split(ratings):

    # We want to take matrix R
    # select ~ 900 users (rows) that have rated more than 10
    # movies and put them into the test set the values we've put should be changed to 0
    # but the row should not be left with 0 ratings!!!
    # all the rows we took out should be predicted



    return train, test


def matrix_factr(R, P, Q, ks, steps=100, gamma=0.0002, lamda=0.02):
    '''
    R: Ratings matrix
    P: |users| * k - user feature matrix
    Q: |movies| * k - movie feature matrix
    ks: number of latent features we want
    steps: iterations
    gamma: learning rate
    lamda: regularization rate
    '''

    Q = Q.T

    for step in tqdm(range(steps)):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i, j] > 0:
                    # Error calculation
                    error_i_j = R[i, j] - np.dot(P[i, :], Q[:, j])

                    for k in range(ks):
                        # Calculate the gradient
                        P[i][k] = P[i][k] + gamma * (2 * error_i_j * Q[k][j] - lamda * P[i][k])
                        Q[k][j] = Q[k][j] + gamma * (2 * error_i_j * P[i][k] - lamda * Q[k][j])

        # Check for convergence
    #    e = 0
    #    for i in range(len(R)):
    #        printProgressBar(i, len(R), suffix=' 2. Convergence')
    #        for j in range(len(R[i])):
    #            if R[i, j] > 0:
    #                e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)

    #                for k in range(ks):
    #                    e = e + (lamda/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
    #    print(f'e = {e}')

    #    if e < 0.001:
    #        break

    return P, Q


def predict_latent_factors(movies, users, ratings, predictions):

    ratings = (movies_description.set_index("movieID")).join(
        ratings.pivot(index='movieID', columns='userID', values='rating'))
    ratings.drop(['year', 'movie'], axis=1, inplace=True)
    ratings = ratings.fillna(0)
    R = ratings.to_numpy()
    # R = ratings_test.to_numpy()

    # First we need to split our data to train and test set!
    trainR, testR = train_test_split(R)

    # n: number of users
    n = len(trainR)
    # m: number of movies
    m = len(trainR[0])
    # k: number of features
    k = 4

    P = np.random.rand(n, k)
    Q = np.random.rand(m, k)

    #u, s, vh = np.linalg.svd(R, full_matrices=False)

    #print(np.shape(u)) # reduce columns 
    #print(np.shape(vh)) #reduce rows
    #Q = u[:, :k]
    #VT = vh[:k, :]
    #s = s[:k]
    #s = np.diag(s)
    #P = np.dot(s, VT)
    #print(np.shape(Q))
    #print(np.shape(P))

    # newP, newQ = matrix_factr(trainR, P, Q, k)

   # newR = np.dot(newP, newQ)
   # number_predictions = len(predictions)
   # result = [[idx+1, newR[predictions.movieID[idx]-1, predictions.userID[idx]-1]] for idx in range(0, number_predictions)]
    result = 0
    return result


#print(predict_latent_factors(movies_description, users_description, ratings_description, predictions_description))


#####
##
# FINAL PREDICTORS
##
#####


def predict_final(movies, users, ratings, predictions):
    # TO COMPLETE

    pass


#####
##
# RANDOM PREDICTORS
# //!!\\ TO CHANGE
##
#####

# By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


#####
##
# SAVE RESULTS
##
#####    


# //!!\\ TO CHANGE by your prediction function
predictions = predict_latent_factors(movies_description, users_description, ratings_description, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)
