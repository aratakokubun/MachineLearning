# coding: utf-8

# Imports
from PRML.Commons.CsvBase import CsvBase

# Class
class WineData(CsvBase):

    '''
    Initialize instance of WineData class with csv row data.
    @param csv_row : each row of csv file
    @raise InvalidCsvException : Csv row data is invalid to create instance.
    '''
    def __init__(self, csv_row):
        CsvBase.__init__(self, csv_row)

    '''
    Apply csv row data to create WineData instance.
    @param csv_row : each row of csv file
    @ref CsvBase.__read_csv_row
    '''
    def read_csv_row(self, csv_row):
        self._class = csv_row[0]
        self._alcohol = csv_row[1]
        self._malic_acid = csv_row[2]
        self._ash = csv_row[3]
        self._alcalinity_of_ash = csv_row[4]
        self._magnesium = csv_row[5]
        self._total_phenols = csv_row[6]
        self._flavanoids = csv_row[7]
        self._nonflavanoid_phenols = csv_row[8]
        self._proanthocyanins = csv_row[9]
        self._color_intensity = csv_row[10]
        self._hue = csv_row[11]
        self._diluted_wines = csv_row[12]
        self._proline = csv_row[13]

    '''
    Validate if data is list and size of it is 14.
    @param csv_row : each row of csv file
    @return true : valid csv row
            false : invalid csv row
    @ref CsvBase.__validate_csv_row
    '''
    @staticmethod
    def validate_csv_row(csv_row):
        return type(csv_row) is list and len(csv_row) == 14

    '''
    Create instance of WineData from csv row data.
    @param cls : class reference
    @param csv_row : each row of csv file
    @return instance of WineData
    @ref CsvBase.instantinate
    '''
    @staticmethod
    def instantinate(csv_row):
        return WineData(csv_row)

    def get_class(self):
        return self._class

    def get_alcohol(self):
        return self._alcohol

    def get_malic_acid(self):
        return self._malic_acid

    def get_ash(self):
        return self._ash

    def get_alcalinity_of_ash(self):
        return self._alcalinity_of_ash

    def get_magnesium(self):
        return self._magnesium

    def get_total_phenols(self):
        return self._total_phenols

    def get_flavanoids(self):
        return self._flavanoids

    def get_nonflavanoid_phenols(self):
        return self._nonflavanoid_phenols

    def get_proanthocyanins(self):
        return self._proanthocyanins

    def get_color_intensity(self):
        return self._color_intensity

    def get_hue(self):
        return self._hue

    def get_diluted_wines(self):
        return self._diluted_wines

    def get_proline(self):
        return self._proline
