import numpy as np
import pandas as pd
import os.path
from random import randint
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import uuid
import time

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
ratings = (movies_description.set_index("movieID")).join(
    ratings_description.pivot(index='movieID', columns='userID', values='rating'))
ratings.drop(['year', 'movie'], axis=1, inplace=True)
ratings = ratings.fillna(0)

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

#-------------------------------------------------Util---------------------------------------------------

def grid_search(samples, R, b, steps=500, gamma=0.01, lamda=0.01, rmse = False, bias = False):
    """
    Perfomrs grid search to find the best pair of hyper paramters k and learning_rate
    """

    ks = [25, 30]
    learn_rates = [0.01, 0.001, 0.0001]

    for k in ks:
        P = np.random.normal(scale=1./k, size=(len(R), k))
        Q = np.random.normal(scale=1./k, size=(len(R[0]), k))
        b_u = np.zeros(len(R))
        b_i = np.zeros(len(R[0]))
        for lr in learn_rates:
            newP, newQ, newB_u, newB_i = matrix_factr(samples, R, P, Q, b, b_u, b_i, gamma=lr, rmse=True, bias=True)
            rmse_score = get_rmse(R, newP, newQ, b, newB_u, newB_i, bias=True)
            with open('grid_search.txt', 'a') as f:
                print(f"k={k}, lr={lr}, rmse={rmse_score}", file=f)


def plot_rmse(mse):
    """
    Plots the mean squared error over all steps
    :param mse:
    :return:
    """
    indeces = [i for i, j in mse]
    mses = [j for i, j in mse]
    with open('mse_bias_100s.txt', 'w') as f:
        print(mses, file=f)
    plt.figure(figsize=((16,4)))
    plt.plot(indeces, mses)
    plt.xticks(indeces, indeces)
    plt.xlabel("Steps")
    plt.ylabel("RMSE")
    plt.grid(axis="y")
    filename = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(filename + '.png')


def get_rmse(R, P, Q, b, b_u, b_i, bias):
    """
    Computes total mean squared error
    :param R:
    :param P:
    :param Q:
    :return:
    """
    i_indeces, j_indeces = R.nonzero()
    pred = predict_all(P, Q, b, b_u, b_i, bias) 
    e = 0
    for i, j in zip(i_indeces, j_indeces):
        e += pow(R[i][j] - pred[i][j], 2)
    return np.sqrt(e/len(i_indeces))


def predict_single(i, j, P, Q, b, b_u, b_i, bias = False):
    """
    Predicts the rating for a single user i and item j
    """
    if bias:
        return (b + b_u[i] + b_i[j] + np.dot(P[i, :], Q[:, j]))
    else:
        return np.dot(P[i, :], Q[:, j])


def predict_all(P, Q, b, b_u, b_i, bias = False):
    """
    Return the whole prediction matrix
    """
    if bias:
        return b + b_u[:, np.newaxis] + b_i[np.newaxis:, ] + np.dot(P, Q)
    else:
        return np.dot(P, Q)


#-----------------------------------------------ALS-----------------------------------------------------------

def als_step(R, update_matrix, fixed_matrix, k, gamma):
    """
    In ALS we update one matrix while the other stays the same
    """
    A = fixed_matrix.T.dot(fixed_matrix) + np.eye(k) * gamma
    b = R.dot(fixed_matrix)
    A_inv = np.linalg.inv(A)
    update_matrix = b.dot(A_inv)

    return update_matrix


def matrix_als(R, P, Q, k, steps=100, gamma=0.01, rmse = False, bias = False):
    """
    Alternating Least Sqaures method for optimizing
    P and Q matrices.
    """
    rmse_data = []
    rmse_old = get_rmse(R, P, Q.T, 0, [], [], False)

    for step in tqdm(range(steps)):
        Q[Q < 0] = 0
        P = als_step(R, P, Q, k, gamma)
        P[P < 0] = 0
        Q = als_step(R.T, Q, P, k, gamma)

        if rmse:
            rmse_new = get_rmse(R, P, Q.T, 0, [], [], False)
            print(rmse_new)
            # Check for convergence and break if new RMSE is bigger than old
            if np.abs(rmse_new - rmse_old) < 0.0001:
                rmse_data.append((step + 1, rmse_new))
                break
            else:
                rmse_data.append((step + 1, rmse_new))
                rmse_old = rmse_new
    if rmse:
        plot_rmse(rmse_data)

    return P, Q


#-----------------------------------------------SDG-----------------------------------------------------------


def matrix_factr(samples, R, P, Q, b, b_u, b_i, steps=200, gamma=0.005, lamda=0.01, rmse = False, bias = False):
    '''
    R: Ratings matrix
    P: |users| * k - user feature matrix
    Q: |movies| * k - movie feature matrix
    ks: number of latent features we want
    steps: iterations
    gamma: learning rate
    lamda: regularization rate - bigger rate means less overfitting
    '''

    Q = Q.T
    rmse_data = []
    rmse_old = get_rmse(R, P, Q, b, b_u, b_i, bias)
    rmse_data.append((0, rmse_old))

    for step in tqdm(range(steps)):
        np.random.shuffle(samples)
        for i, j, r in samples:
            # Error calculation
            e = r - predict_single(i, j, P, Q, b, b_u, b_i, bias)
            # Bias calculation
            if bias:
                b_u[i] += gamma * (e - lamda * b_u[i])
                b_i[j] += gamma * (e - lamda * b_i[j])
            # Gradient calculation
            p_old = P[i, :][:]
            P[i,:] += gamma * (e * Q[:,j] - lamda * P[i,:])
            Q[:,j] += gamma * (e * p_old - lamda * Q[:,j])

        if (step + 1) % 10 == 0: #and rmse:
            rmse_new = get_rmse(R, P, Q, b, b_u, b_i, bias)
            # Check for convergence and break if new RMSE is bigger than old
            if np.abs(rmse_new - rmse_old) < 0.0001:
                rmse_data.append((step + 1, rmse_new))
                break
            else:
                rmse_data.append((step + 1, rmse_new))
                rmse_old = rmse_new
    if rmse:
        plot_rmse(rmse_data)

    return P, Q, b_u, b_i

#-----------------------------------------------Prediction functions for LF-----------------------------------------------------------


def predict_latent_factors_with_bias(movies, users, ratings, predictions):
    # R: Ratings matrix
    R = ratings.to_numpy()
    # n: number of users
    n = len(R)
    # m: number of movies
    m = len(R[0])
    # k: number of features
    k = 5
    # bias: do we want to include bias in our computation
    bias_value = True

    # Initialise random P and Q matrices
    P = np.random.normal(scale=1./k, size=(n, k))
    Q = np.random.normal(scale=1./k, size=(m, k))

    # Initialise biases
    b = np.mean(R[np.where(R != 0)])
    b_u = np.zeros(n)
    b_i = np.zeros(m)

    # Throw away 0 rated entries once instead
    # of checking every step if R[i][j] is 0
    samples = [
        (i, j, R[i, j])
        for i in range(n)
        for j in range(m)
        if R[i][j] > 0
    ]

    newP, newQ, newB_u, newB_i = matrix_factr(samples, R, P, Q, b, b_u, b_i, rmse=False, bias=bias_value)

    newR = predict_all(newP, newQ, b, newB_u, newB_i, bias=bias_value)

    number_predictions = len(predictions)
    result = [[idx+1, newR[predictions.movieID[idx]-1, predictions.userID[idx]-1]] for idx in range(0, number_predictions)]
    return result


def predict_latent_factors_no_bias(movies, users, ratings, predictions):
    # R: Ratings matrix
    R = ratings.to_numpy()
    # n: number of users
    n = len(R)
    # m: number of movies
    m = len(R[0])
    # k: number of features
    k = 25
    # bias: do we want to include bias in our computation
    bias_value = False

    # Initialise random P and Q matrices
    P = np.random.normal(scale=1./k, size=(n, k))
    Q = np.random.normal(scale=1./k, size=(m, k))

    # Initialise biases
    b = np.mean(R[np.where(R != 0)])
    b_u = np.zeros(n)
    b_i = np.zeros(m)

    # Throw away 0 rated entries once instead
    # of checking every step if R[i][j] is 0
    samples = [
        (i, j, R[i, j])
        for i in range(n)
        for j in range(m)
        if R[i][j] > 0
    ]

    newP, newQ, newB_u, newB_i = matrix_factr(samples, R, P, Q, b, b_u, b_i, rmse=True, bias=bias_value)
    newR = predict_all(newP, newQ, b, newB_u, newB_i, bias=bias_value)

    number_predictions = len(predictions)
    result = [[idx+1, newR[predictions.movieID[idx]-1, predictions.userID[idx]-1]] for idx in range(0, number_predictions)]
    return result


def predict_latent_factors_ALS(movies, users, ratings, predictions):
    # R: Ratings matrix
    R = ratings.to_numpy()
    # n: number of users
    n = len(R)
    # m: number of movies
    m = len(R[0])
    # k: number of features
    k = 25
    # bias: do we want to include bias in our computation

    # Initialise random P and Q matrices
    P = np.random.normal(scale=1./k, size=(n, k))
    Q = np.random.normal(scale=1./k, size=(m, k))

    newP, newQ = matrix_als(R, P, Q, k, rmse=True)

    newR = predict_all(newP, newQ.T, b, b_u, b_i)

    number_predictions = len(predictions)
    result = [[idx+1, newR[predictions.movieID[idx]-1, predictions.userID[idx]-1]] for idx in range(0, number_predictions)]
    return result

#####
##
# FINAL PREDICTORS
##
#####


def predict_final(movies, users, ratings, predictions):

    result = predict_latent_factors_with_bias(movies_description, users_description, ratings, predictions_description)

    pass



#####
##
# SAVE RESULTS
##
#####    


# //!!\\ TO CHANGE by your prediction function
predictions = predict_final(movies_description, users_description, ratings, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)
