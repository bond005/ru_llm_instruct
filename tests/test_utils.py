import os
import sys
import unittest

try:
    from utils.utils import levenshtein
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.utils import levenshtein


class TestUtils(unittest.TestCase):
    def test_levenshtein_pos01(self):
        s1 = ['a', 'b', 'c', 'd']
        s2 = ['a', 'b', 'c', 'd']
        self.assertAlmostEqual(levenshtein(s1, s2), 0.0)

    def test_levenshtein_pos02(self):
        s1 = ['a', 'b', 'c', 'd']
        s2 = ['a', 'b', 'b', 'l', 'c', 'd']
        self.assertAlmostEqual(levenshtein(s1, s2), 2.0)

    def test_levenshtein_pos03(self):
        s1 = ['a', 'b', 'c', 'd']
        s2 = ['x', 'y', 'z']
        self.assertAlmostEqual(levenshtein(s1, s2), 4.0)

    def test_levenshtein_neg01(self):
        s1 = []
        s2 = []
        self.assertAlmostEqual(levenshtein(s1, s2), 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
