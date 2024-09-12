import os
import sys
import unittest

import numpy as np
from transformers import LongformerForMaskedLM, LongformerTokenizerFast

try:
    from score.score import calculate_token_embeddings, bert_score
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from score.score import calculate_token_embeddings, bert_score


class TestScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        longformer_name = 'kazzand/ru-longformer-tiny-16384'
        cls.tokenizer = LongformerTokenizerFast.from_pretrained(longformer_name)
        cls.model = LongformerForMaskedLM.from_pretrained(longformer_name)

    def test_calculate_token_embeddings_pos01(self):
        texts = ['Мама мыла раму.', '', 'Папа мыл синхрофазотрон!']
        true_lengths = [len(self.tokenizer.tokenize(cur)) for cur in texts]
        embeddings = calculate_token_embeddings(texts, (self.tokenizer, self.model))
        self.assertEqual(len(embeddings), len(texts))
        for sample_idx in range(len(texts)):
            if true_lengths[sample_idx] > 0:
                self.assertIsInstance(embeddings[sample_idx], np.ndarray)
                self.assertEqual(len(embeddings[sample_idx].shape), 2)
                self.assertEqual(embeddings[sample_idx].shape[0], true_lengths[sample_idx])
                for time_idx in range(true_lengths[sample_idx]):
                    self.assertGreater(np.linalg.norm(x=embeddings[sample_idx][time_idx]), 0)
            else:
                self.assertIsNone(embeddings[sample_idx])

    def test_calculate_token_embeddings_pos02(self):
        texts = ['Мама мыла раму.', '', 'Папа мыл синхрофазотрон!']
        true_lengths = [len(self.tokenizer.tokenize(cur)) for cur in texts]
        embeddings = calculate_token_embeddings(texts, (self.tokenizer, self.model), batch_size=2)
        self.assertEqual(len(embeddings), len(texts))
        for sample_idx in range(len(texts)):
            if true_lengths[sample_idx] > 0:
                self.assertIsInstance(embeddings[sample_idx], np.ndarray)
                self.assertEqual(len(embeddings[sample_idx].shape), 2)
                self.assertEqual(embeddings[sample_idx].shape[0], true_lengths[sample_idx])
                for time_idx in range(true_lengths[sample_idx]):
                    self.assertGreater(np.linalg.norm(x=embeddings[sample_idx][time_idx]), 0)
            else:
                self.assertIsNone(embeddings[sample_idx])

    def test_bert_score_pos01(self):
        references = ['Мама мыла раму.', '', 'Привет, мир!', 'Мама мыла раму.']
        predictions = ['Папа мыл синхрофазотрон', '', '', 'Мама мыла окно.']
        calculated_scores = bert_score(references, predictions, (self.tokenizer, self.model))
        self.assertIsInstance(calculated_scores, list)
        self.assertEqual(len(calculated_scores), len(references))
        for idx in range(len(references)):
            self.assertIsInstance(calculated_scores[idx], float)
            self.assertGreaterEqual(calculated_scores[idx], 0.0)
            self.assertLessEqual(calculated_scores[idx], 1.0)
        self.assertAlmostEqual(calculated_scores[1], 1.0)
        self.assertAlmostEqual(calculated_scores[2], 0.0)
        self.assertGreater(calculated_scores[-1], calculated_scores[0])

    def test_bert_score_neg01(self):
        references = ['Мама мыла раму.', '', 'Привет, мир!', 'Мама мыла раму.']
        predictions = ['Папа мыл синхрофазотрон', '', '']
        true_err_msg = r'The reference texts do not correspond to the predicted texts! 4 != 3'
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = bert_score(references, predictions, (self.tokenizer, self.model))


if __name__ == '__main__':
    unittest.main(verbosity=2)
