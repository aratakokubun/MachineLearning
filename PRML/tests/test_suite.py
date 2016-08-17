# coding: utf-8

import unittest

from PRML.tests.Commons import WineDataTest
from PRML.tests.Commons import CrimeDataTest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromModule(WineDataTest))
    suite.addTest(loader.loadTestsFromModule(CrimeDataTest))

    unittest.TextTestRunner(verbosity=2).run(suite)
