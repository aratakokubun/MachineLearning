# coding: utf-8

class WineData:

    '''
    Initalize wine data with csv row data.
    @param csv_row : row data of csv. Size of the row must be 12.
        csv data from https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    '''
    def __init__(self, csv_row):
        self.__alcohol = csv_row[0]
        self.__malic_acid = csv_row[1]
        self.__ash = csv_row[2]
        self.__alcalinity_of_ash = csv_row[3]
        self.__magnesium = csv_row[4]
        self.__total_phenols = csv_row[5]
        self.__flavanoids = csv_row[6]
        self.__nonflavanoid_phenols = csv_row[7]
        self.__color_intensity = csv_row[8]
        self.__hue = csv_row[9]
        self.__diluted_wines = csv_row[10]
        self.__proline = csv_row[11]

    def get_alcohol(self):
        return self.__alcohol

    def get_malic_acid(self):
        return self.__malic_acid

    def get_ash(self):
        return self.__ash

    def get_alcalinity_of_ash(self):
        return self.__alcalinity_of_ash

    def get_magnesium(self):
        return self.__magnesium

    def get_total_phenols(self):
        return self.__total_phenols

    def get_flavanoids(self):
        return self.__flavanoids

    def get_nonflavanoid_phenols(self):
        return self.__nonflavanoid_phenols

    def get_color_intensity(self):
        return self.__color_intensity

    def get_hue(self):
        return self.__hue

    def get_diluted_wines(self):
        return self.__diluted_wines

    def get_proline(self):
        return self.__proline
