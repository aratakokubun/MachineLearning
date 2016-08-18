# coding: utf-8

# Import libraries
import numpy as np
import csv
import os
import PRML.Commons.DataUtils as DataUtils

# Constants
REGULARIZE_LAMBDA = 0.2
LEARNING_RATE = 0.0002
REGRESSION_LIMIT = 0.0005

# Messages
FMT_PROGRESS = "Count: {0}, weights: {1}, loss: {2}"
FMT_RESULT_CRIME_RATE = "Predicted: {0}, Actual: {1}"

# Utility Class
'''
Handle summarized data of crime data list.
'''
class CrimeDataSummarize:

    def __init__(self, crime_data_list):
        self._data_num = len(crime_data_list)
        self._data_sum = sum([data.get_crime_per_pop() for data in crime_data_list])
        self._unemployed_sum = sum([data.get_unemployed() for data in crime_data_list])
        self._unemployed_square_sum = sum([data.get_unemployed()**2 for data in crime_data_list])
        self._pop_dens_sum = sum([data.get_pop_dens() for data in crime_data_list])
        self._pop_dens_square_sum = sum([data.get_pop_dens()**2 for data in crime_data_list])
        self._data_unemployed_cross_sum = sum([data.get_crime_per_pop()*data.get_unemployed() for data in crime_data_list])
        self._data_pop_dens_cross_sum = sum([data.get_crime_per_pop()*data.get_pop_dens() for data in crime_data_list])
        self._unemployed_pop_dens_cross_sum = sum([data.get_unemployed()*data.get_pop_dens() for data in crime_data_list])

    def get_data_num(self):
        return self._data_num

    def get_data_sum(self):
        return self._data_sum

    def get_unemployed_sum(self):
        return self._unemployed_sum

    def get_unemployed_square_sum(self):
        return self._unemployed_square_sum

    def get_pop_dens_sum(self):
        return self._pop_dens_sum

    def get_pop_dens_square_sum(self):
        return self._pop_dens_square_sum

    def get_data_unemployed_cross_sum(self):
        return self._data_unemployed_cross_sum

    def get_data_pop_dens_cross_sum(self):
        return self._data_pop_dens_cross_sum

    def get_unemplyed_pop_dens_cross_sum(self):
        return self._unemployed_pop_dens_cross_sum

# Functions
'''
Predict crime rate from crime data and weight parameter.
@param crime_data : source crime data to predict crime rate
@param weights : weight parameter for linear regression
 Must be size 3 list [w0, w1, w2].
@return predicted crime rate
'''
def target_func(crime_data, weights):
    return weights[0] + weights[1]*crime_data.get_unemployed() + weights[2]*crime_data.get_pop_dens()

'''
Calculate loss for current weight parameter.
Regularize term is L2 norm.
@param crime_data : list of learning crime data
@param weights : weight parameter for linear regression
@param lam : weight parameter for regularize term
@return loss value
'''
def loss_func(crime_data_list, weights, lam=REGULARIZE_LAMBDA):
    return sum([(crime_data.get_crime_per_pop()-target_func(crime_data, weights))**2 for crime_data in crime_data_list]) + lam*sum([weight**2 for weight in weights[1:]])

'''
Calculate differential for weights
@param weights : weight parameter for linear regression
@param crime_data_summarize : CrimeDataSummarize class object
@param lam : weight parameter for regularize term
@return differential values for weights as [delta_w0, delta_w1, delta_w2]
'''
def diff_func(weights, crime_data_summarize, lam=REGULARIZE_LAMBDA):
    data_num = crime_data_summarize.get_data_num()
    data_sum = crime_data_summarize.get_data_sum()
    unemployed_sum = crime_data_summarize.get_unemployed_sum()
    unemployed_square_sum = crime_data_summarize.get_unemployed_square_sum()
    pop_dens_sum = crime_data_summarize.get_pop_dens_sum()
    pop_dens_square_sum = crime_data_summarize.get_pop_dens_square_sum()
    data_unemplyed_cross_sum = crime_data_summarize.get_data_unemployed_cross_sum()
    data_pop_dens_cross_sum = crime_data_summarize.get_data_pop_dens_cross_sum()
    cross_sum = crime_data_summarize.get_unemplyed_pop_dens_cross_sum()

    # differential for each weight
    slope_w0 = data_num * weights[0] + unemployed_sum * weights[1] + pop_dens_sum * weights[2] - data_sum
    slope_w1 = unemployed_sum * weights[0] + (unemployed_square_sum + lam) * weights[1] + cross_sum * weights[2] - data_unemplyed_cross_sum
    slope_w2 = pop_dens_sum * weights[0] + cross_sum * weights[1] + (pop_dens_square_sum + lam) * weights[2] - data_pop_dens_cross_sum

    # slope to minimize loss function
    delta_w0 = abs(slope_w0) if slope_w0<0 else -abs(slope_w0)
    delta_w1 = abs(slope_w1) if slope_w1<0 else -abs(slope_w1)
    delta_w2 = abs(slope_w2) if slope_w2<0 else -abs(slope_w2)

    return [delta_w0, delta_w1, delta_w2]

'''
Calculate differential for weights
@param weights : weight parameter for linear regression
@param crime_data_summarize : CrimeDataSummarize class object
@param lam : weight parameter for regularize term
@param learning_rate : update each weight with learning_rate * differential for each weight
@return new weights as [w0, w1, w2]
'''
def update_weights(weights, crime_data_summarize, lam=REGULARIZE_LAMBDA, learning_rate=LEARNING_RATE):
    delta_w = diff_func(weights, crime_data_summarize, lam)
    return [weight + learning_rate*delta for weight, delta in zip(weights, delta_w)]

'''
Judge if finish training
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
    crime_data_list = DataUtils.read_crime_data()
    # Instantinate summarize crime data
    summary = CrimeDataSummarize(crime_data_list)
    # Initialize weight parameters
    weights = [0.0, 0.0, 0.0]

    # Training loop
    print()
    count = 0
    print(FMT_PROGRESS.format(count, weights, loss_func(crime_data_list, weights, lam=REGULARIZE_LAMBDA)))
    while True:
        old_weights = list(weights)
        weights = update_weights(weights, summary, lam=REGULARIZE_LAMBDA, learning_rate=LEARNING_RATE)
        if is_finish_training(old_weights, weights, limit=REGRESSION_LIMIT):
            break
        count += 1

        # Print progress
        if count % 5 == 0:
            print(FMT_PROGRESS.format(count, weights, loss_func(crime_data_list, weights, lam=REGULARIZE_LAMBDA)))

    # Draw results
    print(weights)
    for crime_data in crime_data_list:
        predicted_crime_rate = target_func(crime_data, weights)
        # print(FMT_RESULT_CRIME_RATE.format(predicted_crime_rate, crime_data.get_crime_per_pop()))
