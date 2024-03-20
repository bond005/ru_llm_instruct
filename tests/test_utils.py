import os
import sys
import unittest

try:
    from utils.utils import levenshtein, process_multiline
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.utils import levenshtein, process_multiline


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

    def test_process_multiline_pos01(self):
        s = 'abc'
        true_res = 'abc'
        predicted = process_multiline(s)
        self.assertIsInstance(predicted, str)
        self.assertEqual(predicted, true_res)

    def test_process_multiline_pos02(self):
        s = 'abc\r\nd  ef g '
        true_res = ['abc', 'd ef g']
        predicted = process_multiline(s)
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(predicted), len(true_res))
        self.assertEqual(predicted, true_res)

    def test_process_multiline_pos03(self):
        s = '\nabc\n\nd  ef g\n'
        true_res = ['abc', 'd ef g']
        predicted = process_multiline(s)
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(predicted), len(true_res))
        self.assertEqual(predicted, true_res)

    def test_process_multiline_neg01(self):
        s = ''
        true_res = ''
        predicted = process_multiline(s)
        self.assertIsInstance(predicted, str)
        self.assertEqual(predicted, true_res)


if __name__ == '__main__':
    unittest.main(verbosity=2)
