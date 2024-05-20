import os
import sys
import unittest

try:
    from utils.utils import levenshtein, process_multiline, process_target, tokenize_text
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.utils import levenshtein, process_multiline, process_target, tokenize_text


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

    def test_process_target_pos01(self):
        s = ' abc\n'
        self.assertEqual(process_target(s), 'abc')

    def test_process_target_pos02(self):
        s = 'abc </s> '
        self.assertEqual(process_target(s), 'abc')

    def test_process_target_pos03(self):
        s = 'ab\nc</s>'
        self.assertEqual(process_target(s), 'ab\nc')

    def test_process_target_neg01(self):
        s = ''
        self.assertEqual(process_target(s), '')

    def test_tokenize_pos01(self):
        s = 'Тем временем, сообщает «Новый Регион — Екатеринбург», 1 февраля 2001 года.'
        true_words = ['Тем', 'временем', ',', 'сообщает', '«', 'Новый', 'Регион', '—', 'Екатеринбург', '»', ',',
                      '1', 'февраля', '2001', 'года', '.']
        predicted = tokenize_text(s)
        self.assertIsInstance(predicted, list)
        for idx, val in enumerate(predicted):
            self.assertIsInstance(val, str, msg=f'The {idx} word {val} has a wrong type!')
        self.assertEqual(len(predicted), len(true_words), msg=f'{predicted}')
        self.assertEqual(predicted, true_words, msg=f'{predicted}')

    def test_tokenize_pos02(self):
        s = 'A. I. Galushkin graduated from the Bauman Moscow Higher Technical School in 1963.'
        true_words = ['A', '.', 'I', '.', 'Galushkin', 'graduated', 'from', 'the', 'Bauman', 'Moscow', 'Higher',
                      'Technical', 'School', 'in', '1963', '.']
        predicted = tokenize_text(s)
        self.assertIsInstance(predicted, list)
        for idx, val in enumerate(predicted):
            self.assertIsInstance(val, str, msg=f'The {idx} word {val} has a wrong type!')
        self.assertEqual(len(predicted), len(true_words))
        self.assertEqual(predicted, true_words)


if __name__ == '__main__':
    unittest.main(verbosity=2)
