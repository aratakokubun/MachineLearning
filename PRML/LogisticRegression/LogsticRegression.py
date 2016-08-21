# coding: utf-8

import numpy as np
import math
import PRML.Commons.DataUtils as DataUtils
import random

# Constants
REGULARIZE_LAMBDA = 0.01
LEARNING_RATE = 0.009
REGRESSION_LIMIT = 0.001

# Messages
FMT_PROGRESS = "Count: {0}, weights: {1}, loss: {2}"
FMT_RESULT_CRIME_RATE = "Predicted: {0}, Actual: {1}. Probability for class 1: {2}"
FMT_RESULT_CORRECTNESS = "Correct {0}/{1}"

# Functions
'''
Predict wine class evaluation from wine data and weight parameter.
@param wine_data : source wine data to predict wine class
@param weights : weight parameter for linear regression
 Must be size 14 list [w0, w1, ...].
@return predicted wine class evaluation
'''
def target_func(wine_data, weights):
    return weights[0] + np.dot(weights[1:], wine_data[1:].T)

'''
Return logistic sigmoid value for input.
@param x : input value
@return logistic sigmoid of x
'''
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

'''
Predict the probability to be class 1 (+1) from wine data and weight parameter.
The score of class 1 is +1.
The score of class 2 is -1.
@param wine_data : source wine data to predict wine class
@param weights : weight parameter for linear regression
 Must be size 14 list [w0, w1, ...].
@return predicted wine class
'''
def calc_class_prob(wine_data, weights):
    return sigmoid(target_func(wine_data, weights))

'''
Calculate loss for current weight parameter.
Regularize term is L2 norm.
@param wine_data_list : list of wine data to predict wine class
@param weights : weight parameter for linear regression
 Must be size 14 list [w0, w1, ...].
@param lam : weight parameter for regularize term
@return loss value
'''
def loss_func(wine_data_list, weights, lam=REGULARIZE_LAMBDA):
    norm_term_fc = lambda wine_data: math.log(1 + math.exp(-wine_data[0] * target_func(wine_data, weights)))
    return np.apply_along_axis(norm_term_fc, 1, wine_data_list).sum() + lam * np.sum(weights[1:]**2)

'''
Calculate gradient descent for weights to update weights parameter.
@param wine_data_list : List of wine data
@param weights : weight parameter for linear regression
@param lam : weight parameter for regularize term
@return differential values for weights as [delta_w0, delta_w1, ...]
'''
def gradient_descent(wine_data_list, weights, lam=REGULARIZE_LAMBDA):
    delta_w_list = np.array([])

    for wine_data in wine_data_list:
        arrayed_data = np.hstack((1, wine_data[1:]))
        wine_class = wine_data[0]

        exp_term = math.exp(- wine_class * target_func(wine_data, weights))
        grad_norm = lambda x: - wine_class * x * exp_term / (1 + exp_term)
        grad_norm_vec = np.vectorize(grad_norm)
        np.append(delta_w_list, grad_norm_vec(arrayed_data), axis=0)

    delta_w = delta_w_list.sum(axis=0) + np.hstack((0, 2 * lam * weights[1:]))
    descent = lambda x: abs(x) if x<0 else -abs(x)
    descent_veg = np.vectorize(descent)
    return descent_veg(delta_w)

def stochastic_gradient_descent(sample_wine_data, weights, lam=REGULARIZE_LAMBDA):
    wine_class = sample_wine_data[0]
    exp_term = math.exp(- wine_class * target_func(sample_wine_data, weights))
    grad_norm = lambda x: - wine_class * x * exp_term / (1 + exp_term)
    grad_norm_vec = np.vectorize(grad_norm)
    arrayed_data = np.hstack((1, sample_wine_data[1:]))
    delta_w = grad_norm_vec(arrayed_data) + np.hstack((0, 2 * lam * weights[1:]))
    descent = lambda x: abs(x) if x<0 else -abs(x)
    descent_veg = np.vectorize(descent)
    return descent_veg(delta_w)

'''
Update weigts parameter to decrease loss value for prediction.
@param wine_data_list : List of wine data
@param weights : weight parameter for linear regression
@param lam : weight parameter for regularize term
@param learning_rate : update each weight with learning_rate * differential for each weight
@return new weights as [w0, w1, ...]
'''
def update_weights(wine_data_list, weights, lam=REGULARIZE_LAMBDA, learning_rate=LEARNING_RATE):
    # delta_w = gradient_descent(wine_data_list, weights, lam)
    selected_wine_data = random.choice(wine_data_list)
    delta_w = stochastic_gradient_descent(selected_wine_data, weights, lam)
    return weights + learning_rate*delta_w

'''
Judge if finish training.
@param old_weights
@param new_weights
@param limit : boarder value to finish training
'''
def is_finish_training(old_weights, new_weights, limit=REGRESSION_LIMIT):
    diff_term = np.linalg.norm(np.array(new_weights)-np.array(old_weights))
    normalize_term = np.linalg.norm(np.array(old_weights))
    diff = diff_term / (normalize_term if normalize_term>0 else 0.000001)
    return diff < limit

# Main
if __name__ == "__main__":
    # Read learning data
    wine_data_list = DataUtils.read_wine_data()
    # Extract class data 1 and 2
    # then, convert class 1 to +1, 2 to -1
    # normalize data other than class from 0.0 to 1.0
    new_wine_data_list = []
    max_wine_data = np.array([wine_data.to_np_array() for wine_data in wine_data_list]).max(axis=0)
    for wine_data in wine_data_list:
        wine_class = wine_data.get_class()
        if wine_class in (1, 2):
            new_class = +1 if wine_class==1 else -1
            wine_data_norm = wine_data.to_np_array() / max_wine_data
            wine_data_norm[0] = new_class
            new_wine_data_list.append(wine_data_norm)
    wine_data_list = new_wine_data_list
    # Initialize weight parameters
    # Random value from -1.0 to 1.0
    weights = np.random.rand(14) * 2.0 - 1.0

    # Training loop
    count = 0
    print(FMT_PROGRESS.format(count, weights, loss_func(wine_data_list, weights, lam=REGULARIZE_LAMBDA)))
    while True:
        old_weights = list(weights)
        weights = update_weights(wine_data_list, weights, lam=REGULARIZE_LAMBDA, learning_rate=LEARNING_RATE)
        if is_finish_training(old_weights, weights, limit=REGRESSION_LIMIT):
            break
        count += 1

        # Print progress
        if count % 20 == 0:
            print(FMT_PROGRESS.format(count, weights, loss_func(wine_data_list, weights, lam=REGULARIZE_LAMBDA)))

    # Draw results
    print(weights)
    count_correct_class = 0
    for wine_data in wine_data_list:
        actual_class = wine_data[0]
        prob_class_1 = calc_class_prob(wine_data, weights)
        predicted_class = 1 if prob_class_1 > 0.5 else -1
        print(FMT_RESULT_CRIME_RATE.format(predicted_class, actual_class, prob_class_1))

        if predicted_class*actual_class > 0:
            count_correct_class += 1

    print(FMT_RESULT_CORRECTNESS.format(count_correct_class, len(wine_data_list)))
