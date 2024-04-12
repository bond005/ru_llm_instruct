import os
import sys
import unittest

try:
    from ner.ner import find_subphrase, find_entities_in_text
    from ner.factrueval import load_sample
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ner.ner import find_subphrase, find_entities_in_text
    from ner.factrueval import load_sample


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

    def test_find_subphrase_pos05(self):
        sentence = ['Левичев', 'стал', 'кандидатом', 'от', 'эсеров', 'на', 'выборах', 'мэра', 'Москвы']
        subphrase = ['Москвы']
        self.assertEqual(find_subphrase(sentence, subphrase), 8)

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
        entities = ['Russian Academy of Sciences', 'Academy</s>']
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

    def test_find_entities_in_text_pos04(self):
        source_text = 'Левичев стал кандидатом от эсеров на выборах мэра Москвы'
        entities = ['Москвы</s>']
        entity_class = 'LOCATION'
        true_res = [
            ('Левичев', 'O'),
            ('стал', 'O'),
            ('кандидатом', 'O'),
            ('от', 'O'),
            ('эсеров', 'O'),
            ('на', 'O'),
            ('выборах', 'O'),
            ('мэра', 'O'),
            ('Москвы', 'B-LOCATION')
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

    def test_find_entities_in_text_neg02(self):
        source_text = 'A. I. Galushkin graduated from the Bauman Moscow Higher Technical School in 1963.'
        entities = ['Bauman Higher Technical School']
        entity_class = 'ORGANIZATION'
        with self.assertRaises(ValueError):
            _ = find_entities_in_text(source_text, entities, entity_class, raise_exception=True)


class TestFactRuEval(unittest.TestCase):
    def test_load_sample_pos01(self):
        dataset_path = os.path.join(os.path.dirname(__file__), 'testdata')
        dataset_item = 'book_100'
        true_text = ('В понедельник 28 июня у здания мэрии Москвы на Тверской площади состоялась очередная '
                     'несанкционированная акция протеста «День гнева», в этот раз направленная, главным образом, '
                     'против политики московских и подмосковных властей. Среди требований, выдвигаемых организаторами '
                     'акции: «немедленная отставка мэра Москвы Юрия Лужкова, расследование итогов его деятельности», '
                     '«созыв московского общественного форума для обсуждения путей реформирования основных сфер '
                     'жизнедеятельности в Москве», «восстановление прямых выборов глав регионов России», «роспуск '
                     'нелегитимной Мосгордумы», отставка подмосковного губернатора Бориса Громова и др. Участникам '
                     'акции предлагалось принести с собой лист бумаги или кусок ткани чёрного цвета, символизирующие '
                     '«чёрную метку» для Юрия Лужкова.\n\nНачало акции было намечено на 19 часов; подчёркивалось, что '
                     'она состоится несмотря на запрет властей. Освещающие акцию блоггеры сообщили, что автобусы с '
                     'милицией стали занимать площадь у памятника Юрию Долгорукому ещё с 15 часов дня, центральная '
                     'часть площади была огорожена. Ко времени начала акции вокруг огороженной территории собралось '
                     'множество журналистов и прохожих, по мере прибытия самих участников акции милиция начала '
                     'планомерно их задерживать и заталкивать в автобусы.\n\nВсего, по сообщениям блоггеров и СМИ, '
                     'было задержано более 30 человек. Пользователь ЖЖ zyalt сообщает, что среди задержанных оказался '
                     'депутат муниципального собрания района Отрадное, сопредседатель московского отделения движения '
                     '«Солидарность» Михаил Вельмакин, известный правозащитник Лев Пономарев, координатор движения '
                     '«Левый фронт» Сергей Удальцов.\n\nОрганизаторами акции выступили движение «За права человека», '
                     'Союз координационных советов (СКС), институт «Коллективное действие», «Жилищная солидарность», '
                     'Движение общежитий Москвы, Движение в защиту Химкинского леса, движение «Московский совет», '
                     '«Координационный совет пострадавших соинвесторов».')
        true_entities = {
            'organization': [
                (31, 43),
                (562, 572),
                (1395, 1434),
                (1473, 1496),
                (1451, 1496),
                (1566, 1588),
                (1647, 1666),
                (1668, 1696),
                (1698, 1701),
                (1704, 1736),
                (1738, 1761),
                (1763, 1788),
                (1790, 1824),
                (1826, 1853),
                (1855, 1904)
            ],
            'location': [
                (37, 43),
                (47, 63),
                (306, 312),
                (477, 483),
                (531, 537),
                (958, 984),
                (1808, 1824)
            ],
            'person': [
                
            ]
        }
        loaded_text, loaded_entities = load_sample(dataset_path, dataset_item)


if __name__ == '__main__':
    unittest.main(verbosity=2)
