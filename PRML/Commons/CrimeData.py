# coding: utf-8

# Imports
from PRML.Commons.CsvBase import CsvBase

# Class
class CrimeData(CsvBase):

    '''
    Initialize instance of CrimeData class with csv row data.
    @param csv_row : each row of csv file
    @raise InvalidCsvException : Csv row data is invalid to create instance.
    '''
    def __init__(self, csv_row):
        CsvBase.__init__(self, csv_row)

    '''
    Apply csv row data to create CrimeData instance.
    @param csv_row : each row of csv file
    @ref CsvBase.__read_csv_row
    '''
    def read_csv_row(self, csv_row):
        self._unemployed = float(csv_row[37])
        self._pop_dens = float(csv_row[119])
        self._crime_per_pop = float(csv_row[127])

    '''
    Validate if data is list and size of it is 128.
    @param csv_row : each row of csv file
    @return true : valid csv row
            false : invalid csv row
    @ref CsvBase.__validate_csv_row
    '''
    @staticmethod
    def validate_csv_row(csv_row):
        return type(csv_row) is list and len(csv_row) == 128

    '''
    Create instance of CrimeData from csv row data.
    @param cls : class reference
    @param csv_row : each row of csv file
    @return instance of CrimeData
    @ref CsvBase.instantinate
    '''
    @staticmethod
    def instantinate(csv_row):
        return CrimeData(csv_row)

    def get_unemployed(self):
        return self._unemployed

    def get_pop_dens(self):
        return self._pop_dens

    def get_crime_per_pop(self):
        return self._crime_per_pop
