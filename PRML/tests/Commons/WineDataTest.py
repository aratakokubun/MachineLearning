# coding: utf-8

# Imports
import unittest
from PRML.Commons.WineData import WineData

# Class
class WineDataTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_validate_csv_row(self):
        test_data = [WineDataValidateFixture([0 for _ in range(14)], True),
        WineDataValidateFixture([0 for _ in range(15)], False),
        WineDataValidateFixture([0 for _ in range(13)], False),]

        for data in test_data:
            self.assertEqual(WineData.validate_csv_row(data.get_csv_row()), data.get_is_valid())

'''
Supply required data for wine data test
'''
class WineDataValidateFixture():

    def __init__(self, csv_row, is_valid):
        self._csv_row = csv_row
        self._is_valid = is_valid

    def get_csv_row(self):
        return self._csv_row

    def get_is_valid(self):
        return self._is_valid

if __name__ == '__main__':
    unittest.main()
