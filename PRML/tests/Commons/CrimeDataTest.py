# coding: utf-8

# Import libraries
import unittest
from PRML.Commons.CrimeData import CrimeData

# Class
class CrimeDataTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_validate_csv_row(self):
        test_data = [CrimeDataValidateFixture([0 for _ in range(128)], True),
        CrimeDataValidateFixture([0 for _ in range(129)], False),
        CrimeDataValidateFixture([0 for _ in range(127)], False),]

        for data in test_data:
            self.assertEqual(CrimeData.validate_csv_row(data.get_csv_row()), data.get_is_valid())

'''
Supply required data for wine data test
'''
class CrimeDataValidateFixture():

    def __init__(self, csv_row, is_valid):
        self._csv_row = csv_row
        self._is_valid = is_valid

    def get_csv_row(self):
        return self._csv_row

    def get_is_valid(self):
        return self._is_valid

if __name__ == '__main__':
    unittest.main()
