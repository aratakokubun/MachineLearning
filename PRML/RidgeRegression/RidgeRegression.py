# coding: utf-8

# Import libraries
import numpy as np
import csv
import os
import PRML.Commons.DataUtils as DataUtils

# Constants
REGULARIZE_LAMBDA = 0.1
LEARNING_RATE = 0.1
REGRESSION_LIMIT = 0.1

# Messages
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

    def get_pop_dens_square(self):
        return self._pop_dens_square_sum

    def get_cross_sum(self):
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
    return sum([(crime_data.get_crime_per_pop()-target_func(crime_data, weights))**2 for crime_data in crime_data_list]) +
     lam*sum([weight**2 for weight in weights[1:]])

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
    unemployed_pop_dens_cross_sum = crime_data_summarize.get_unemployed_pop_dens_cross_sum()

    # differential for each weight
    target_w0 = (data_sum - unemployed_sum*weights[1] - pop_dens_sum*weights[2]) / crime_data_num
    delat_w0  = target_w0 - weights[0]
    target_w1 = (data_sum - unemployed_sum*weights[0] - unemployed_pop_dens_cross_sum*weights[2]) / (unemployed_square_sum + lam)
    delat_w1  = target_w1 - weights[1]
    target_w2 = (data_sum - pop_dens_sum*weights[0] - unemployed_pop_dens_cross_sum*weights[1]) / (pop_dens_square_sum + lam)
    delat_w2  = target_w2 - weights[2]

    return [delat_w0, delat_w1, delat_w2]

'''
Calculate differential for weights
@param weights : weight parameter for linear regression
 weights are overwriten in this function.
@param crime_data_summarize : CrimeDataSummarize class object
@param lam : weight parameter for regularize term
@param learning_rate : update each weight with learning_rate * differential for each weight
'''
def update_weights(weights, crime_data_summarize, lam=REGULARIZE_LAMBDA, learning_rate=LEARNING_RATE):
    delta_w = diff_func(weights, crime_data_summarize, lam)
    weights = [weight - lam*delta for weight, delta in zip(weights, delta_w)]

'''
Judge if finish training
@param old_weights
@param new_weights
@param limit : boarder value to finish training
'''
def is_finish_training(old_weights, new_weights, limit=REGRESSION_LIMIT):
    diff = sum([w**2 for w in (new_weights-old_weights)]) / sum(old_weights)
    return diff < limit

# Main
if __name__ == "__main__":
    # Read learning data
    crime_data_list = DataUtils.read_wine_data()
    # Instantinate summarize crime data
    summary = CrimeDataSummarize(crime_data_list)
    # Initialize weight parameters
    weights = [0.0, 0.0, 0.0]

    # Training loop
    count = 0
    while True:
        old_weights = list(weights)
        update_weights(weights, summary, lam=REGULARIZE_LAMBDA, learning_rate=LEARNING_RATE)
        if is_finish_training(old_weights, weights, limit=REGRESSION_LIMIT):
            break
        count += 1

    # Draw results
    for crime_data in crime_data_list:
        predicted_crime_rate = target_func(crime_data, weights)
        print(FMT_RESULT_CRIME_RATE.format(predicted_crime_rate, crime_data.get_crime_per_pop()))
