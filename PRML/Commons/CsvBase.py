# coding: utf-8

# Imports
from abc import ABCMeta, abstractmethod
import csv
import sys

# Constants
MSG_NOT_IMPLEMENTED = "Not implemented method {0}"

# Classes
'''
Base class to create list of instances from csv file
'''
class CsvBase(object):
    __metaclass__ = ABCMeta

    '''
    Initialize instance of each csv data class with csv row data.
    @param csv_row : each row of csv file
    @raise InvalidCsvException : Csv row data is invalid to create instance.
    '''
    def __init__(self, csv_row):
        if self.validate_csv_row(csv_row):
            self.read_csv_row(csv_row)
        else:
            raise InvalidCsvException(csv_row)

    '''
    Validate csv row data to create instance.
    @param csv_row : each row of csv file
    @return true : valid csv row
            false : invalid csv row
    '''
    @staticmethod
    @abstractmethod
    def validate_csv_row(csv_row):
        print(MSG_NOT_IMPLEMENTED.format(sys._getframe().f_code.co_name))

    '''
    Apply csv row data to create instance.
    @param csv_row : each row of csv file
    '''
    @abstractmethod
    def read_csv_row(self, csv_row):
        print(MSG_NOT_IMPLEMENTED.format(sys._getframe().f_code.co_name))

    '''
    Read csv data to create list of instances for each row.
    @param cls : class reference
    @param csv_path : path of csv file to read
    @raise InvalidCsvException : Csv row data is invalid to create instance.
    '''
    @classmethod
    def read_csv_data(cls, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            return [cls.instantinate(row) for row in reader]

    '''
    Create instance of this class from csv row data.
    @param cls : class reference
    @param csv_row : each row of csv file
    @return instance of implemented class of CsvBase
    '''
    @staticmethod
    @abstractmethod
    def instantinate(csv_row):
        print(MSG_NOT_IMPLEMENTED.format(sys._getframe().f_code.co_name))

'''
Exception caused by reading invalid csv data
'''
class InvalidCsvException(Exception):

    def __init__(self, row_data):
        self.row_data = row_data

    def __str__(self):
        print(row_data)
