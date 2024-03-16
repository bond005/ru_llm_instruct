import os
import sys
import unittest

try:
    from ner.ner import find_subphrase, find_entities_in_text
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ner.ner import find_subphrase, find_entities_in_text


class TestNER(unittest.TestCase):
    def test_find_subphrase_neg01(self):
        sentence = ['a', 'b', 'c']
        subprhase = ['a', 'b', 'c', 'd']
        self.assertEqual(find_subphrase(sentence, subprhase), -1)

    def test_find_subphrase_neg02(self):
        sentence = ['a', 'b', 'c', 'd', 'e']
        subprhase = ['c', 'b']
        self.assertEqual(find_subphrase(sentence, subprhase), -1)

    def test_find_subphrase_neg03(self):
        sentence = ['a', 'b', 'c', 'e', 'd']
        subprhase = ['a', 'b', 'c', 'd', 'e']
        self.assertEqual(find_subphrase(sentence, subprhase), -1)

    def test_find_subphrase_pos01(self):
        sentence = ['a', 'b', 'c', 'd', 'e']
        subprhase = ['b', 'c']
        self.assertEqual(find_subphrase(sentence, subprhase), 1)

    def test_find_subphrase_pos02(self):
        sentence = ['a', 'b', 'c', 'd', 'e']
        subprhase = ['c']
        self.assertEqual(find_subphrase(sentence, subprhase), 2)

    def test_find_subphrase_pos03(self):
        sentence = ['a', 'b', 'c', 'd', 'e']
        subprhase = ['a', 'b', 'c', 'd', 'e']
        self.assertEqual(find_subphrase(sentence, subprhase), 0)

    def test_find_subphrase_pos04(self):
        sentence = ['a', 'B', 'c', 'd', 'e']
        subprhase = ['b', 'C']
        self.assertEqual(find_subphrase(sentence, subprhase), 1)

    def test_find_entities_in_text_pos01(self):
        source_text = 'A. I. Galushkin graduated from the Bauman Moscow Higher Technical School in 1963.'
        entities = ['A. I. Galushkin']
        entity_class = 'PERSON'
        true_res = [
            ('A', 'B-PERSON'),
            ('.', 'I-PERSON'),
            ('I', 'I-PERSON'),
            ('.', 'I-PERSON'),
            ('Galushkin', 'I-PERSON'),
            ('graduated', 'O'),
            ('from', 'O'),
            ('the', 'O'),
            ('Bauman', 'O'),
            ('Moscow', 'O'),
            ('Higher', 'O'),
            ('Technical', 'O'),
            ('School', 'O'),
            ('in', 'O'),
            ('1963', 'O'),
            ('.', 'O')
        ]
        predicted = find_entities_in_text(source_text, entities, entity_class)
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(predicted), len(true_res))
        for cur in predicted:
            self.assertIsInstance(cur, tuple)
            self.assertEqual(len(cur), 2)
            self.assertIsInstance(cur[0], str)
            self.assertIsInstance(cur[1], str)
        self.assertEqual(predicted, true_res)

    def test_find_entities_in_text_pos02(self):
        source_text = 'A. I. Galushkin graduated from the Bauman Moscow Higher Technical School in 1963.'
        entities = ['Bauman Moscow Higher Technical School']
        entity_class = 'ORGANIZATION'
        true_res = [
            ('A', 'O'),
            ('.', 'O'),
            ('I', 'O'),
            ('.', 'O'),
            ('Galushkin', 'O'),
            ('graduated', 'O'),
            ('from', 'O'),
            ('the', 'O'),
            ('Bauman', 'B-ORGANIZATION'),
            ('Moscow', 'I-ORGANIZATION'),
            ('Higher', 'I-ORGANIZATION'),
            ('Technical', 'I-ORGANIZATION'),
            ('School', 'I-ORGANIZATION'),
            ('in', 'O'),
            ('1963', 'O'),
            ('.', 'O')
        ]
        predicted = find_entities_in_text(source_text, entities, entity_class)
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(predicted), len(true_res))
        for cur in predicted:
            self.assertIsInstance(cur, tuple)
            self.assertEqual(len(cur), 2)
            self.assertIsInstance(cur[0], str)
            self.assertIsInstance(cur[1], str)
        self.assertEqual(predicted, true_res)

    def test_find_entities_in_text_pos03(self):
        source_text = ('K.V. Vorontsov was awarded the medal of the Russian Academy of Sciences and '
                       'the prize for young scientists in recognition of his winning the competition held '
                       'to commemorate the 275th anniversary of the Academy (1999).')
        entities = ['Russian Academy of Sciences', 'Academy']
        entity_class = 'ORGANIZATION'
        true_res = [
            ('K', 'O'),
            ('.', 'O'),
            ('V', 'O'),
            ('.', 'O'),
            ('Vorontsov', 'O'),
            ('was', 'O'),
            ('awarded', 'O'),
            ('the', 'O'),
            ('medal', 'O'),
            ('of', 'O'),
            ('the', 'O'),
            ('Russian', 'B-ORGANIZATION'),
            ('Academy', 'I-ORGANIZATION'),
            ('of', 'I-ORGANIZATION'),
            ('Sciences', 'I-ORGANIZATION'),
            ('and', 'O'),
            ('the', 'O'),
            ('prize', 'O'),
            ('for', 'O'),
            ('young', 'O'),
            ('scientists', 'O'),
            ('in', 'O'),
            ('recognition', 'O'),
            ('of', 'O'),
            ('his', 'O'),
            ('winning', 'O'),
            ('the', 'O'),
            ('competition', 'O'),
            ('held', 'O'),
            ('to', 'O'),
            ('commemorate', 'O'),
            ('the', 'O'),
            ('275th', 'O'),
            ('anniversary', 'O'),
            ('of', 'O'),
            ('the', 'O'),
            ('Academy', 'B-ORGANIZATION'),
            ('(', 'O'),
            ('1999', 'O'),
            (').', 'O')
        ]
        predicted = find_entities_in_text(source_text, entities, entity_class)
        self.assertIsInstance(predicted, list, msg=f'{predicted}')
        self.assertEqual(len(predicted), len(true_res), msg=f'{predicted}')
        for cur in predicted:
            self.assertIsInstance(cur, tuple)
            self.assertEqual(len(cur), 2)
            self.assertIsInstance(cur[0], str)
            self.assertIsInstance(cur[1], str)
        self.assertEqual(predicted, true_res)

    def test_find_entities_in_text_neg01(self):
        source_text = 'A. I. Galushkin graduated from the Bauman Moscow Higher Technical School in 1963.'
        entities = ['Bauman Higher Technical School']
        entity_class = 'ORGANIZATION'
        true_res = [
            ('A', 'O'),
            ('.', 'O'),
            ('I', 'O'),
            ('.', 'O'),
            ('Galushkin', 'O'),
            ('graduated', 'O'),
            ('from', 'O'),
            ('the', 'O'),
            ('Bauman', 'O'),
            ('Moscow', 'O'),
            ('Higher', 'O'),
            ('Technical', 'O'),
            ('School', 'O'),
            ('in', 'O'),
            ('1963', 'O'),
            ('.', 'O')
        ]
        predicted = find_entities_in_text(source_text, entities, entity_class)
        self.assertIsInstance(predicted, list)
        self.assertEqual(len(predicted), len(true_res))
        for cur in predicted:
            self.assertIsInstance(cur, tuple)
            self.assertEqual(len(cur), 2)
            self.assertIsInstance(cur[0], str)
            self.assertIsInstance(cur[1], str)
        self.assertEqual(predicted, true_res)


if __name__ == '__main__':
    unittest.main(verbosity=2)
