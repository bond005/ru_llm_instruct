import os
import sys
import unittest

try:
    from ner.ner import find_subphrase, find_entities_in_text
    from ner.factrueval import load_sample, join_nested_entities, load_spans, extend_entity_bounds
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ner.ner import find_subphrase, find_entities_in_text
    from ner.factrueval import load_sample, join_nested_entities, load_spans, extend_entity_bounds


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
    def test_load_spans(self):
        src_text = ('В понедельник 28 июня у здания мэрии Москвы на Тверской площади состоялась очередная '
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
        true_spans = {
            22763: (37, 43),  # 6
            22764: (31, 36),  # 5
            22765: (47, 55),  # 8
            22766: (56, 63),  # 7
            22767: (313, 317),  # 4
            22768: (318,  325),  # 7
            22769: (306, 312),  # 6
            22770: (477, 483),  # 6
            22771: (531, 537),  # 6
            22772: (562, 572),  # 10
            22773: (610, 616),  # 6
            22774: (617, 624),  # 7
            22775: (756, 760),  # 4
            22776: (761, 768),  # 7
            22777: (301, 305),  # 4
            22778: (598, 609),  # 11
            22779: (584, 609),  # 25
            68733: (584, 597),  # 13
            22782: (968, 972),  # 4
            22783: (973, 984),  # 11
            22784: (958, 967),  # 9
            22785: (1340, 1345),  # 5
            22788: (1426, 1434),  # 8
            22789: (1419, 1425),  # 6
            22790: (1395, 1418),  # 23
            22793: (1483, 1495),  # 12
            22794: (1473, 1481),  # 8
            22795: (1451, 1472),  # 21
            22796: (1497, 1503),  # 6
            22797: (1504, 1513),  # 9
            22798: (1539, 1542),  # 3
            22799: (1543, 1552),  # 9
            22800: (1576, 1587),  # 11
            22801: (1566, 1574),  # 8
            22802: (1589, 1595),  # 6
            22803: (1596, 1604),  # 8
            22804: (1387, 1394),  # 7
            22805: (1436, 1450),  # 14
            22806: (1525, 1538),  # 13
            22807: (1554, 1565),  # 11
            22808: (1463, 1472),  # 9
            22809: (1648, 1665),  # 17
            22810: (1638, 1646),  # 8
            22811: (1668, 1696),  # 28
            22812: (1668, 1672),  # 4
            22813: (1698, 1701),  # 3
            22814: (1714, 1735),  # 21
            22815: (1704, 1712),  # 8
            22816: (1739, 1760),  # 21
            22817: (1763, 1788),  # 25
            22818: (1763, 1771),  # 8
            22819: (1790, 1824),  # 34
            22820: (1790, 1798),  # 8
            22821: (1836, 1852),  # 16
            22822: (1826, 1834),  # 8
            22823: (1856, 1903),  # 47
            22824: (1872, 1877),  # 5
            22825: (1856, 1877),  # 21
            38271: (1808, 1824),  # 16
            38272: (1820, 1824),  # 4
        }
        loaded_spans = load_spans(src_text, os.path.join(os.path.dirname(__file__), 'testdata', 'book_100.spans'))
        self.assertIsInstance(loaded_spans, dict)
        self.assertEqual(set(loaded_spans.keys()), set(true_spans.keys()))
        for k in sorted(list(true_spans.keys())):
            self.assertEqual(loaded_spans[k], true_spans[k])

    def test_load_sample(self):
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
                (1451, 1496),
                (1566, 1588),
                (1638, 1666),
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
                (584, 597),
                (958, 984),
                (1419, 1434),
                (1808, 1824)
            ],
            'person': [
                (313,  325),
                (610, 624),
                (756, 768),
                (968, 984),
                (1340, 1345),
                (1497, 1513),
                (1539, 1552),
                (1589, 1604)
            ]
        }
        res = load_sample(dataset_path, dataset_item)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        loaded_text, loaded_entities = res
        self.assertIsInstance(loaded_text, str)
        self.assertIsInstance(loaded_entities, dict)
        self.assertEqual(loaded_text, true_text)
        self.assertEqual(set(loaded_entities.keys()), set(true_entities.keys()))
        for entity_type in sorted(list(true_entities.keys())):
            self.assertIsInstance(loaded_entities[entity_type], list)
            self.assertEqual(len(loaded_entities[entity_type]), len(true_entities[entity_type]))
            for i, v in enumerate(loaded_entities[entity_type]):
                self.assertIsInstance(v, tuple)
                self.assertEqual(len(v), 2)
                self.assertEqual(v, true_entities[entity_type][i])

    def test_join_nested_entities_pos01(self):
        src = [(0, 10), (10, 15), (25, 45)]
        self.assertEqual(join_nested_entities(src), src)

    def test_join_nested_entities_pos02(self):
        src = [(0, 10)]
        self.assertEqual(join_nested_entities(src), src)

    def test_join_nested_entities_pos03(self):
        src = []
        self.assertEqual(join_nested_entities(src), src)

    def test_join_nested_entities_pos04(self):
        src = [(0, 10), (10, 15), (25, 45), (8, 11)]
        tgt = [(0, 15), (25, 45)]
        self.assertEqual(join_nested_entities(src), tgt)

    def test_join_nested_entities_pos05(self):
        src = [(0, 10), (10, 15), (25, 45), (8, 11), (14, 27)]
        tgt = [(0, 45)]
        self.assertEqual(join_nested_entities(src), tgt)

    def test_join_nested_entities_pos09(self):
        src = [(0, 10), (10, 15), (14, 48), (25, 45), (8, 11)]
        tgt = [(0, 48)]
        self.assertEqual(join_nested_entities(src), tgt)

    def test_extend_entity_bounds_pos01(self):
        s = 'координатор движения «Левый фронт» Сергей Удальцов'
        src_bounds = (22, 33)
        true_bounds = (21, 34)
        self.assertEqual(extend_entity_bounds(s, *src_bounds), true_bounds)

    def test_extend_entity_bounds_pos02(self):
        s = 'координатор движения «Левый фронт» Сергей Удальцов'
        src_bounds = (21, 33)
        true_bounds = (21, 34)
        self.assertEqual(extend_entity_bounds(s, *src_bounds), true_bounds)

    def test_extend_entity_bounds_pos03(self):
        s = 'координатор движения «Левый фронт» Сергей Удальцов'
        src_bounds = (22, 34)
        true_bounds = (21, 34)
        self.assertEqual(extend_entity_bounds(s, *src_bounds), true_bounds)

    def test_extend_entity_bounds_pos04(self):
        s = 'координатор движения «Левый фронт» Сергей Удальцов'
        src_bounds = (21, 34)
        true_bounds = (21, 34)
        self.assertEqual(extend_entity_bounds(s, *src_bounds), true_bounds)

    def test_extend_entity_bounds_pos05(self):
        s = 'координатор движения Левый фронт Сергей Удальцов'
        src_bounds = (21, 32)
        true_bounds = (21, 32)
        self.assertEqual(extend_entity_bounds(s, *src_bounds), true_bounds)


if __name__ == '__main__':
    unittest.main(verbosity=2)
