import numpy as np
import pandas as pd

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

#####
##
# COLLABORATIVE FILTERING
##
##
#####

#-------------------------------------------------User-Based CF---------------------------------------------------

def predict_collaborative_filtering_useritem(ratings, predictions, similarity):
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
    # Idea from https://github.com/john-x-jiang/Recommending-System/blob/master/src/similarity_functions.py

    # Pearson correlation
    similarity_matrix = None

    if(similarity == 1):
        similarity_matrix = np.corrcoef(ratings_matrix)

    # Cosine for cosine and adj cosine similarities
    def cosine(a, b):
        # Vector multiplication
        p = a*b
        # Norm multiplication
        c = p / b
        d = p / a
        norm = np.linalg.norm(c) * np.linalg.norm(d)
        # Division by 0
        if not norm:
            return 0
        # Return cos
        return np.dot(a, b) / norm

    # # Cosine similarity
    def cosineMatrix(ratings):
        no_users = np.shape(ratings)[1]
        sim = np.zeros((no_users, no_users))
        # Simple cosine similarity
        for i in range(no_users):
            for j in range(i, no_users):
                sim[i][j] = cosine(ratings[:, i], ratings[:, j])
                sim[j][i] = sim[i][j]
        return sim

    if(similarity == 2):
        similarity_matrix = cosineMatrix(ratings_matrix.T+1e-9)


    # Adjusted cosine similarity
    def adjustedCosineMatrix(ratings):
        no_users = np.shape(ratings)[0]
        no_items = np.shape(ratings)[1]
        sim = np.zeros((no_items, no_items))
        # Calculate the average user rating to adjust for it
        user_avg = np.zeros(no_users)
        zero_items = 0
        # Take care of users that didnt rate items
        for u in range(no_users):
            for i in range(no_items):
                if (ratings[u, i] == 0):
                    zero_items += 1
                user_avg[u] += ratings[u, i]
            user_avg[u] /= zero_items
        # Calculate the similarity adjusting to user avg
        for i in range(no_items):
            for j in range(i, no_items):
                sim[i][j] = cosine(ratings[:, i] - user_avg,
                                   ratings[:, j] - user_avg)
                sim[j][i] = sim[i][j]
        return sim

    if(similarity_matrix is None):
        similarity_matrix = adjustedCosineMatrix(ratings_matrix.T)

    # 3. Compute predictions

    # Function 1
    def prediction_scoref1 (user, movie):
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
                sum_numer += similarity_matrix[user, idx]*(user_ratings[movie])
                sum_denomin += similarity_matrix[user, idx]

        return sum_numer/sum_denomin


    # Function 2
    # # Another approach might be possible:
    # # Global baseline hybrid

    # # Get the average rating of each user
    avgs = ratings_description.groupby('userID').agg(['mean'])
    avgs.drop(['movieID'], axis=1, inplace=True)
    user_rating_avgs = avgs.to_numpy()

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

    def prediction_scoref2 (user, movie):
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

    # Function 3
    # Predict the score of unknown rating. Function given in the link above
    def prediction_scoref3 (user, movie):
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


    # Creating the final predictions format
    number_predictions = len(predictions)
    # 3 predictions for each of the predictor functions.
    final_predictionsf1 = [[idx+1, prediction_scoref1(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in range(0, number_predictions )]
    final_predictionsf2 = [[idx+1, prediction_scoref2(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in range(0, number_predictions )]
    final_predictionsf3 = [[idx+1, prediction_scoref3(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in range(0, number_predictions )]

    return final_predictionsf1, final_predictionsf2, final_predictionsf3

#-------------------------------------------------Item-Based CF---------------------------------------------------

def predict_collaborative_filtering_itemitem(ratings, predictions, similarity):
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
    similarity_matrix = None
    # Pearson correlation
    if(similarity == 1):
        similarity_matrix = np.corrcoef(ratings_matrix.T + 1e-9)

    # Cosine similarity first attempt
    # similarity_matrix = ratings_matrix.T.dot(ratings_matrix) + 1e-9
    # norms = np.array([np.sqrt(np.diagonal(similarity_matrix))])
    # similarity_matrix = (similarity_matrix / norms / norms.T)

    # # Adjusted Cosine similarity first attempt
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
    # for idx, itemA in enumerate(ratings_transpose):
    #     for idy, itemB in enumerate(ratings_transpose):
    #         adj_cosine_similarity_matrix[idx][idy] = adj_cosine_similarity(idx, idy)
    #         adj_cosine_similarity_matrix[idy][idx] = adj_cosine_similarity_matrix[idx][idy]

    # Cosine for cosine and adj cosine similarities (idea from https://github.com/john-x-jiang/Recommending-System/blob/master/src/similarity_functions.py)
    def cosine(a, b):
        # Vector multiplication
        p = a*b
        # Norm multiplication
        c = p / b
        d = p / a
        norm = np.linalg.norm(c) * np.linalg.norm(d)
        # Division by 0
        if not norm:
            return 0
        # Return cos
        return np.dot(a, b) / norm

    # Cosine similarity matrix
    def cosineMatrix(ratings):
        no_items = np.shape(ratings)[1]
        sim = np.zeros((no_items, no_items))
        # Simple cosine similarity
        for i in range(no_items):
            for j in range(i, no_items):
                sim[i][j] = cosine(ratings[:, i], ratings[:, j])
                sim[j][i] = sim[i][j]
        return sim

    if(similarity == 2):
        similarity_matrix = cosineMatrix(ratings_matrix +1e-9)

    # Adjusted cosine similarity matrix
    def adjustedCosineMatrix(ratings):
        no_items = np.shape(ratings)[0]
        no_users = np.shape(ratings)[1]
        sim = np.zeros((no_users, no_users))
        # Calculate average item rating to adjust for it
        item_avg_rating = np.zeros(no_items)
        zero_items = 0
        # Take care of items that were not rated by users
        for u in range(no_items):
            for i in range(no_users):
                if (ratings[u, i] == 0):
                    zero_items += 1
                item_avg_rating[u] += ratings[u, i]
            item_avg_rating[u] /= zero_items
        # Calculate the similarity adjusting to item avg
        for i in range(no_users):
            for j in range(i, no_users):
                sim[i][j] = cosine(ratings[:, i] - item_avg_rating,
                                   ratings[:, j] - item_avg_rating)
                sim[j][i] = sim[i][j]
        return sim

    if(similarity_matrix is None):
        similarity_matrix = adjustedCosineMatrix(ratings_matrix)

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

    # Function 2 (Global Biases)
    def prediction_scoref2 (user, movie):
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

    # Function 3
    def prediction_scoref3 (user, movie):
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


    # Creating the final predictions format
    # number_predictions = len(predictions)
    # final_predictionsf1 = [[idx+1, prediction_scoref1(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in range(0, number_predictions )]
    # final_predictionsf2 = [[idx+1, prediction_scoref2(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in range(0, number_predictions )]
    # final_predictionsf3 = [[idx+1, prediction_scoref3(predictions.userID[idx]-1, predictions.movieID[idx]-1)] for idx in range(0, number_predictions )]
    #
    # return final_predictionsf1, final_predictionsf2, final_predictionsf3


    # Enabling only the best-performing CF for recommender model combiner
    number_predictions = len(predictions)
    best_prediction_cf = [prediction_scoref3(predictions.userID[idx]-1, predictions.movieID[idx]-1)[0] for idx in range(0, number_predictions )]
    return best_prediction_cf


#####
##
# LATENT FACTORS
##
#####

ratings = (movies_description.set_index("movieID")).join(
    ratings_description.pivot(index='movieID', columns='userID', values='rating'))
ratings.drop(['year', 'movie'], axis=1, inplace=True)
ratings = ratings.fillna(0)

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

    for step in range(steps):
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

    for step in range(steps):
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

    return P, Q, b_u, b_i

def lf_svd(ratings, predictions):
    # R: Ratings matrix
    R = ratings.to_numpy()
    # k: number of features
    k = 5

    b = np.mean(R[np.where(R != 0)])

    print("Running SVD ... ")
    u, s, vh = np.linalg.svd(R, full_matrices=False)

    s = np.diag(s)
    s = s[0:k, 0:k]
    u = u[:, 0:k]
    vh = vh[0:k, :]

    root = np.sqrt(s)
 
    US = np.dot(u, root)
    sV = np.dot(root, vh)
    newR = np.dot(US, sV)

    newR += b

    number_predictions = len(predictions)
    result = [[idx+1, newR[predictions.movieID[idx]-1, predictions.userID[idx]-1]] for idx in range(0, number_predictions)]
    return result


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
    # result = [[idx+1, newR[predictions.movieID[idx]-1, predictions.userID[idx]-1]] for idx in range(0, number_predictions)]
    # return result

    # For combiner
    result = [newR[predictions.movieID[idx]-1, predictions.userID[idx]-1] for idx in range(0, number_predictions)]
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

    newR = predict_all(newP, newQ.T, 0, [], [])

    number_predictions = len(predictions)
    result = [[idx+1, newR[predictions.movieID[idx]-1, predictions.userID[idx]-1]] for idx in range(0, number_predictions)]
    return result


#####
##
# FINAL, BEST CF AND LF PREDICTORS
##
#####

def predict_CF_final():
    result = predict_collaborative_filtering_itemitem(ratings_description, predictions_description, 1)

    return result

def predict_LF_final():

    result = predict_latent_factors_with_bias(movies_description, users_description, ratings, predictions_description)

    return result

#####
##
# PREDICTOR COMBINER
##
#####

def combine_CF_LF():
    CF_final_prediction = predict_CF_final()
    LF_final_prediction = predict_LF_final()

    final_combined = (np.asarray(CF_final_prediction) + np.asarray(LF_final_prediction))/2

    number_predictions = len(predictions_description)
    result = [[idx + 1, final_combined[idx]] for idx in range(0, number_predictions)]
    return result

#####
##
# SAVE RESULTS
##
#####

predictions = combine_CF_LF()

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)
