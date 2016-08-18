# coding: utf-8

# Import libraries
import numpy as np
import csv
import os
from PRML.Commons.WineData import WineData as Wd
from PRML.Commons.CrimeData import CrimeData as Cd

# Constants
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')

# Wine Data
# from https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
WINE_DATA = os.path.join(DATA_DIR, 'wine.csv')
def read_wine_data(csv_file_path = WINE_DATA):
    return Wd.read_csv_data(csv_file_path)

# Community and Crimes
# from http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data
CRIME_DATA = os.path.join(DATA_DIR, 'crime.csv')
def read_crime_data(csv_file_path=CRIME_DATA):
    return Cd.read_csv_data(csv_file_path)
