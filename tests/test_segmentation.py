import os
import sys
import unittest

try:
    from segmentation.segmentation import load_samples
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from segmentation.segmentation import load_samples


class TestSegmentation(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
