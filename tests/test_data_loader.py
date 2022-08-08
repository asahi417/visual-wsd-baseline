""" UnitTest """
import unittest
import logging
import json
from vwsd import data_loader

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):

    def test(self):
        d = data_loader()
        logging.info(json.dumps(d, indent=4))


if __name__ == "__main__":
    unittest.main()
