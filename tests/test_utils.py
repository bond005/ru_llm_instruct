import os
import sys
import unittest

import spacy

try:
    from utils.utils import levenshtein, process_multiline, process_target, tokenize_text
    from utils.utils import split_long_text, split_text_by_sentences
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.utils import levenshtein, process_multiline, process_target, tokenize_text
    from utils.utils import split_long_text, split_text_by_sentences


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

    def test_split_text_by_sentences_pos01(self):
        s = ('Что такое data science? Это наука об автоматическом выявлении закономерностей в эмпирических данных. '
             'А что такое Open Data Science? Это крупнейшее в России сообщество фанатов — профессионалов и '
             'любителей — науки о данных, машинного обучения и нейронных сетей! Вот уже много лет сообщество '
             'Open Data Science проводит конференцию DataFest — праздник для всех, кто занимается алгоритмами и '
             'системами машинного обучения — от самых начинающих до матёрых профессионалов. В этом году DataFest '
             'проходит в смешанном онлайно-офлайновом формате с 25 мая до 2 июня.')
        true_sentences = [
            (0, 5),
            (5, 15),
            (15, 22),
            (22, 43),
            (43, 74),
            (74, 92)
        ]
        true_tokens = [
            (0, 3),
            (4, 9),
            (10, 14),
            (15, 22),
            (22, 23),
            (24, 27),
            (28, 33),
            (34, 36),
            (37, 51),
            (52, 61),
            (62, 77),
            (78, 79),
            (80, 92),
            (93, 99),
            (99, 100),
            (101, 102),
            (103, 106),
            (107, 112),
            (113, 117),
            (118, 122),
            (123, 130),
            (130, 131),
            (132, 135),
            (136, 146),
            (147, 148),
            (149, 155),
            (156, 166),
            (167, 174),
            (175, 176),
            (177, 191),
            (192, 193),
            (194, 203),
            (204, 205),
            (206, 211),
            (212, 213),
            (214, 220),
            (220, 221),
            (222, 231),
            (232, 240),
            (241, 242),
            (243, 252),
            (253, 258),
            (258, 259),
            (260, 263),
            (264, 267),
            (268, 273),
            (274, 277),
            (278, 288),
            (289, 293),
            (294, 298),
            (299, 306),
            (307, 315),
            (316, 327),
            (328, 336),
            (337, 338),
            (339, 347),
            (348, 351),
            (352, 356),
            (356, 357),
            (358, 361),
            (362, 372),
            (373, 384),
            (385, 386),
            (387, 396),
            (397, 406),
            (407, 415),
            (416, 417),
            (418, 420),
            (421, 426),
            (427, 437),
            (438, 440),
            (441, 448),
            (449, 463),
            (463, 464),
            (465, 466),
            (467, 471),
            (472, 476),
            (477, 485),
            (486, 494),
            (495, 496),
            (497, 506),
            (507, 514),
            (514, 515),
            (515, 525),
            (526, 533),
            (534, 535),
            (536, 538),
            (539, 542),
            (543, 545),
            (546, 547),
            (548, 552),
            (552, 553)
        ]
        predicted = split_text_by_sentences(s, spacy.load('ru_core_news_sm'))
        self.assertIsInstance(predicted, tuple)
        self.assertEqual(len(predicted), 2)
        self.assertIsInstance(predicted[0], list)
        self.assertIsInstance(predicted[1], list)
        self.assertEqual(predicted[0], true_sentences)
        self.assertEqual(predicted[1], true_tokens)

    def test_split_long_text_pos01(self):
        s = ('Что такое data science? Это наука об автоматическом выявлении закономерностей в эмпирических данных. '
             'А что такое Open Data Science? Это крупнейшее в России сообщество фанатов — профессионалов и '
             'любителей — науки о данных, машинного обучения и нейронных сетей! Вот уже много лет сообщество '
             'Open Data Science проводит конференцию DataFest — праздник для всех, кто занимается алгоритмами и '
             'системами машинного обучения — от самых начинающих до матёрых профессионалов. В этом году DataFest '
             'проходит в смешанном онлайно-офлайновом формате с 25 мая до 2 июня.')
        true_texts = [
            ('Что такое data science? Это наука об автоматическом выявлении закономерностей в эмпирических данных. '
             'А что такое Open Data Science? Это крупнейшее в России сообщество фанатов — профессионалов и '
             'любителей — науки о данных, машинного обучения и нейронных сетей! Вот уже много лет сообщество '
             'Open Data Science проводит конференцию DataFest — праздник для всех, кто занимается алгоритмами и '
             'системами машинного обучения — от самых начинающих до матёрых профессионалов. В этом году DataFest '
             'проходит в смешанном онлайно-офлайновом формате с 25 мая до 2 июня.')
        ]
        predicted_texts = split_long_text(s, 600, spacy.load('ru_core_news_sm'))
        self.assertIsInstance(predicted_texts, list)
        self.assertEqual(len(predicted_texts), len(true_texts))
        for idx, val in enumerate(predicted_texts):
            self.assertIsInstance(val, tuple)
            self.assertEqual(len(val), 2)
            self.assertIsInstance(val[0], int)
            self.assertIsInstance(val[1], int)
            self.assertGreater(val[1], val[0])
            self.assertGreaterEqual(val[0], 0)
            self.assertEqual(s[val[0]:val[1]].strip(), true_texts[idx])

    def test_split_long_text_pos02(self):
        s = ('Что такое data science? Это наука об автоматическом выявлении закономерностей в эмпирических данных. '
             'А что такое Open Data Science? Это крупнейшее в России сообщество фанатов — профессионалов и '
             'любителей — науки о данных, машинного обучения и нейронных сетей! Вот уже много лет сообщество '
             'Open Data Science проводит конференцию DataFest — праздник для всех, кто занимается алгоритмами и '
             'системами машинного обучения — от самых начинающих до матёрых профессионалов. В этом году DataFest '
             'проходит в смешанном онлайно-офлайновом формате с 25 мая до 2 июня.')
        true_texts = [
            ('Что такое data science? Это наука об автоматическом выявлении закономерностей в эмпирических данных. '
             'А что такое Open Data Science?'),
            ('Это крупнейшее в России сообщество фанатов — профессионалов и любителей — науки о данных, машинного '
             'обучения и нейронных сетей! Вот уже много лет сообщество Open Data Science проводит конференцию '
             'DataFest — праздник для всех, кто занимается алгоритмами и системами машинного обучения — от самых '
             'начинающих до матёрых профессионалов. В этом году DataFest проходит в смешанном онлайно-офлайновом '
             'формате с 25 мая до 2 июня.')
        ]
        predicted_texts = split_long_text(s, 500, spacy.load('ru_core_news_sm'))
        self.assertIsInstance(predicted_texts, list)
        self.assertEqual(len(predicted_texts), len(true_texts))
        for idx, val in enumerate(predicted_texts):
            self.assertIsInstance(val, tuple)
            self.assertEqual(len(val), 2)
            self.assertIsInstance(val[0], int)
            self.assertIsInstance(val[1], int)
            self.assertGreater(val[1], val[0])
            self.assertGreaterEqual(val[0], 0)
            self.assertEqual(s[val[0]:val[1]].strip(), true_texts[idx], msg=s[val[0]:val[1]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
