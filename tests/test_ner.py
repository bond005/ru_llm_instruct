import os
import sys
import unittest

try:
    from ner.ner import find_subphrase, find_entities_in_text, calculate_entity_bounds, match_entities_to_tokens
    from ner.factrueval import load_sample, join_nested_entities, load_spans, extend_entity_bounds
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ner.ner import find_subphrase, find_entities_in_text, calculate_entity_bounds, match_entities_to_tokens
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
            (')', 'O'),
            ('.', 'O')
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

    def test_find_entities_in_text_pos05(self):
        source_text = ('1 февраля на ступеньках будущего президентского центра наследия возле бизнес-центра «Демидов» '
                       'открылся памятник первому президенту России Борису Николаевичу Ельцину. Об этом человеке '
                       'говорят разное, особенно на Урале, где его знали не только как президента, но и как '
                       'специалиста в строительной отрасли, первого секретаря Свердловского обкома КПСС. '
                       'Но отрицать его огромные заслуги невозможно. Уральцы никогда не забудут того, что сделал '
                       'Борис Николаевич для Свердловской области, ещё будучи первым секретарём обкома партии. '
                       'По словам свердловского губернатора Александра Мишарина, «уже тогда для него не существовало '
                       'понятия „невыполнимо“, он жил другими категориями и всех вокруг себя заряжал уверенностью».\n\n'
                       'В Екатеринбурге много связано с именем Бориса Николаевича: в его честь названа улица, '
                       'Уральский центр, его имя носит университет, ежегодно в уральской столице на кубок по волейболу '
                       'имени первого российского президента собираются лучшие мировые команды. '
                       'А теперь память о нём будет увековечена в мраморе. Президентский центр наследия ещё '
                       'не достроен, но десятиметровая мраморная стела-обелиск с барельефом украсила город уже сегодня.'
                       '\n\nСкульптор Георгий Франгулян в 2007 году соорудил памятный мотив в виде флага на могиле '
                       'Бориса Ельцина, его работа очень понравилась супруге первого президента — Наине Иосифовне, и '
                       'создание обелиска тоже поручили этому художнику. По мнению скульптора, мрамор — наилучший '
                       'материал для воплощения его замысла. Было принято решение сделать его не в бронзе, не '
                       'в граните, а в мраморе, мрамор живой материал, полупрозрачный, он очень хорошо сочетается с '
                       'нашим климатом. «Это такая глыба в движении, это глыба, которой и был Борис Николаевич '
                       'Ельцин», — говорит скульптор.\n\nСпециально на торжества, посвященные юбилею Бориса '
                       'Николаевича в Екатеринбург прибыл президент РФ Дмитрий Медведев. Сегодня утром, открывая '
                       'памятник, он отметил, что «Россия должна быть благодарна Ельцину за то, что в самый сложный '
                       'период страна не свернула с пути изменений, провела серьёзные преобразования и сегодня '
                       'движется вперёд». В церемонии также приняли участие вдова Ельцина Наина Иосифовна, '
                       'его друзья, представители федеральной власти, глава Свердловской области Александр Мишарин, '
                       'руководители соседних регионов.\n\nБорис Николаевич сделал больше чем кто бы то ни было для '
                       'создания правового государства, гражданского общества в нашей стране. Мы помним, какую '
                       'значительную роль в создании Конституции сыграл первый президент России. И в дни празднования '
                       'юбилея Бориса Николаевича в городе проходит первый Форум общенациональной программы '
                       '«Гражданское общество — модернизация России», на котором обсуждаются важнейшие вопросы. '
                       'Вчера на форуме работали секции, посвящённые средствам массовой информации, судебной и '
                       'военной реформам, модернизации экономики, материнству и защите детства, экологии и другим не '
                       'менее важным проблемам нашей действительности. Гражданское общество — это общество, в котором '
                       'власть нацелена на поддержку тех общественных институтов, которые в конечном итоге делают более'
                       'комфортной жизнь человека. И на Урале такие институты активно создаются. В регионе первым '
                       'в России возник институт Уполномоченного по правам человека. В конце 80-х, начале 90-х годов '
                       'прошлого века в Свердловске появилась городская дискуссионная трибуна — один из первых '
                       'российских, как выражались тогда, «рупоров гласности». И заложил основу этому первый '
                       'президент, живший тогда ещё в Свердловской области. В мае 1981 года, во Дворце молодёжи, '
                       'он провёл знаковую встречу со студентами и преподавателями 16 свердловских вузов. В 1982 году '
                       'появились регулярные телевизионные передачи-разговоры Ельцина с жителями области. Это было '
                       'начало строительства гражданского общества в не самую, казалось бы, подходящую эпоху.\n\n'
                       'Тем временем, сообщает «Новый Регион — Екатеринбург», 1 февраля 2001 года, в день 80-й '
                       'годовщины со дня рождения Бориса Николаевича, уличные художники проведут в Екатеринбурге '
                       'праздничную акцию и откроют «альтернативный памятник Ельцину».')
        entities = ['президентского центра наследия', 'Свердловского обкома КПСС', 'обкома партии',
                    'Уральский центр', 'Президентский центр наследия', '«Новый Регион — Екатеринбург»']
        entity_class = 'ORGANIZATION'
        predicted = find_entities_in_text(source_text, entities, entity_class, raise_exception=True)
        true_entity_tags = {'O', 'B-ORGANIZATION', 'I-ORGANIZATION'}
        predicted_entity_tags = set()
        for cur in predicted:
            self.assertIsInstance(cur, tuple)
            self.assertEqual(len(cur), 2)
            self.assertIsInstance(cur[0], str)
            self.assertIsInstance(cur[1], str)
            self.assertIn(cur[1], true_entity_tags)
            predicted_entity_tags.add(cur[1])
        self.assertEqual(predicted_entity_tags, true_entity_tags)

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

    def test_calculate_entity_bounds_pos01(self):
        source_text = ('1 февраля на ступеньках будущего президентского центра наследия возле бизнес-центра «Демидов» '
                       'открылся памятник первому президенту России Борису Николаевичу Ельцину. Об этом человеке '
                       'говорят разное, особенно на Урале, где его знали не только как президента, но и как '
                       'специалиста в строительной отрасли, первого секретаря Свердловского обкома КПСС. '
                       'Но отрицать его огромные заслуги невозможно. Уральцы никогда не забудут того, что сделал '
                       'Борис Николаевич для Свердловской области, ещё будучи первым секретарём обкома партии. '
                       'По словам свердловского губернатора Александра Мишарина, «уже тогда для него не существовало '
                       'понятия „невыполнимо“, он жил другими категориями и всех вокруг себя заряжал уверенностью».\n\n'
                       'В Екатеринбурге много связано с именем Бориса Николаевича: в его честь названа улица, '
                       'Уральский центр, его имя носит университет, ежегодно в уральской столице на кубок по волейболу '
                       'имени первого российского президента собираются лучшие мировые команды. '
                       'А теперь память о нём будет увековечена в мраморе. Президентский центр наследия ещё '
                       'не достроен, но десятиметровая мраморная стела-обелиск с барельефом украсила город уже сегодня.'
                       '\n\nСкульптор Георгий Франгулян в 2007 году соорудил памятный мотив в виде флага на могиле '
                       'Бориса Ельцина, его работа очень понравилась супруге первого президента — Наине Иосифовне, и '
                       'создание обелиска тоже поручили этому художнику. По мнению скульптора, мрамор — наилучший '
                       'материал для воплощения его замысла. Было принято решение сделать его не в бронзе, не '
                       'в граните, а в мраморе, мрамор живой материал, полупрозрачный, он очень хорошо сочетается с '
                       'нашим климатом. «Это такая глыба в движении, это глыба, которой и был Борис Николаевич '
                       'Ельцин», — говорит скульптор.\n\nСпециально на торжества, посвященные юбилею Бориса '
                       'Николаевича в Екатеринбург прибыл президент РФ Дмитрий Медведев. Сегодня утром, открывая '
                       'памятник, он отметил, что «Россия должна быть благодарна Ельцину за то, что в самый сложный '
                       'период страна не свернула с пути изменений, провела серьёзные преобразования и сегодня '
                       'движется вперёд». В церемонии также приняли участие вдова Ельцина Наина Иосифовна, '
                       'его друзья, представители федеральной власти, глава Свердловской области Александр Мишарин, '
                       'руководители соседних регионов.\n\nБорис Николаевич сделал больше чем кто бы то ни было для '
                       'создания правового государства, гражданского общества в нашей стране. Мы помним, какую '
                       'значительную роль в создании Конституции сыграл первый президент России. И в дни празднования '
                       'юбилея Бориса Николаевича в городе проходит первый Форум общенациональной программы '
                       '«Гражданское общество — модернизация России», на котором обсуждаются важнейшие вопросы. '
                       'Вчера на форуме работали секции, посвящённые средствам массовой информации, судебной и '
                       'военной реформам, модернизации экономики, материнству и защите детства, экологии и другим не '
                       'менее важным проблемам нашей действительности. Гражданское общество — это общество, в котором '
                       'власть нацелена на поддержку тех общественных институтов, которые в конечном итоге делают более'
                       'комфортной жизнь человека. И на Урале такие институты активно создаются. В регионе первым '
                       'в России возник институт Уполномоченного по правам человека. В конце 80-х, начале 90-х годов '
                       'прошлого века в Свердловске появилась городская дискуссионная трибуна — один из первых '
                       'российских, как выражались тогда, «рупоров гласности». И заложил основу этому первый '
                       'президент, живший тогда ещё в Свердловской области. В мае 1981 года, во Дворце молодёжи, '
                       'он провёл знаковую встречу со студентами и преподавателями 16 свердловских вузов. В 1982 году '
                       'появились регулярные телевизионные передачи-разговоры Ельцина с жителями области. Это было '
                       'начало строительства гражданского общества в не самую, казалось бы, подходящую эпоху.\n\n'
                       'Тем временем, сообщает «Новый Регион — Екатеринбург», 1 февраля 2001 года, в день 80-й '
                       'годовщины со дня рождения Бориса Николаевича, уличные художники проведут в Екатеринбурге '
                       'праздничную акцию и откроют «альтернативный памятник Ельцину».')
        entities = ['президентского центра наследия', 'Свердловского обкома КПСС', 'обкома партии',
                    'Уральский центр', 'Президентский центр наследия', '«Новый Регион — Екатеринбург»']
        true_entity_bounds = [
            (33, 63),
            (321, 346),
            (509, 522),
            (796, 811),
            (1014, 1042),
            (3755, 3784)
        ]
        predicted_bounds = calculate_entity_bounds(source_text, entities)
        self.assertIsInstance(predicted_bounds, list)
        self.assertEqual(len(predicted_bounds), len(true_entity_bounds))
        for idx in range(len(true_entity_bounds)):
            self.assertIsInstance(predicted_bounds[idx], tuple, msg=f'Entity {idx} has incorrect type!')
            self.assertEqual(len(predicted_bounds[idx]), 2, msg=f'Entity {idx} has incorrect length!')
            self.assertEqual(predicted_bounds[idx],true_entity_bounds[idx], msg=f'Entity {idx} is wrong!')

    def test_calculate_entity_bounds_pos02(self):
        source_text = ('В числе претендентов на место Саутера называли высшего чиновника министерства юстиции '
                       'Елену Каган (Elena Kagan) и судью апелляционного суда Дайан Вуд (Diane Wood).')
        entities = ['министерства юстиции', 'министерства', 'апелляционного суда']
        true_entity_bounds = [
            (65, 85),
            (65, 77),
            (120, 139)
        ]
        predicted_bounds = calculate_entity_bounds(source_text, entities)
        self.assertIsInstance(predicted_bounds, list)
        self.assertEqual(len(predicted_bounds), len(true_entity_bounds))
        for idx in range(len(true_entity_bounds)):
            self.assertIsInstance(predicted_bounds[idx], tuple, msg=f'Entity {idx} has incorrect type!')
            self.assertEqual(len(predicted_bounds[idx]), 2, msg=f'Entity {idx} has incorrect length!')
            self.assertEqual(predicted_bounds[idx],true_entity_bounds[idx], msg=f'Entity {idx} is wrong!')

    def test_calculate_entity_bounds_pos03(self):
        source_text = 'В этом тексте нет именованных сущностей такого типа.'
        entities = ['министерства юстиции', 'министерства', 'апелляционного суда']
        predicted_bounds = calculate_entity_bounds(source_text, entities)
        self.assertIsInstance(predicted_bounds, list)
        self.assertEqual(len(predicted_bounds), 0)

    def test_match_entities_to_tokens_pos01(self):
        tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        entities = [['b', 'c'], ['f']]
        true_entity_bounds = [(1, 3), (5, 6)]
        true_penalty = 0
        predicted_variants = match_entities_to_tokens(tokens, entities, [], 0)
        self.assertIsInstance(predicted_variants, list)
        self.assertGreater(len(predicted_variants), 0)
        for cur_variant in predicted_variants:
            self.assertIsInstance(cur_variant, tuple, msg=f'{cur_variant}')
            self.assertEqual(len(cur_variant), 2, msg=f'{cur_variant}')
            self.assertIsInstance(cur_variant[0], list, msg=f'{cur_variant}')
            self.assertIsInstance(cur_variant[1], int, msg=f'{cur_variant}')
            self.assertGreaterEqual(cur_variant[1], 0, msg=f'{cur_variant}')
            for it in cur_variant[0]:
                self.assertIsInstance(it, tuple, msg=f'{cur_variant}')
                self.assertEqual(len(it), 2, msg=f'{cur_variant}')
                self.assertIsInstance(it[0], int, msg=f'{cur_variant}')
                self.assertIsInstance(it[1], int, msg=f'{cur_variant}')
                self.assertLess(it[0], it[1], msg=f'{cur_variant}')
                self.assertGreaterEqual(it[0], 0, msg=f'{cur_variant}')
        predicted_variants.sort(key=lambda it: it[1])
        self.assertEqual(predicted_variants[0][1], true_penalty, msg=f'{predicted_variants[0]}')
        self.assertEqual(predicted_variants[0][0], true_entity_bounds, msg=f'{predicted_variants[0]}')

    def test_match_entities_to_tokens_pos02(self):
        tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        entities = [['b', 'c'], ['c'], ['f']]
        true_entity_bounds = [(1, 3), (2, 3), (5, 6)]
        true_penalty = 0
        predicted_variants = match_entities_to_tokens(tokens, entities, [], 0)
        self.assertIsInstance(predicted_variants, list)
        self.assertGreater(len(predicted_variants), 0)
        for cur_variant in predicted_variants:
            self.assertIsInstance(cur_variant, tuple, msg=f'{cur_variant}')
            self.assertEqual(len(cur_variant), 2, msg=f'{cur_variant}')
            self.assertIsInstance(cur_variant[0], list, msg=f'{cur_variant}')
            self.assertIsInstance(cur_variant[1], int, msg=f'{cur_variant}')
            self.assertGreaterEqual(cur_variant[1], 0, msg=f'{cur_variant}')
            for it in cur_variant[0]:
                self.assertIsInstance(it, tuple, msg=f'{cur_variant}')
                self.assertEqual(len(it), 2, msg=f'{cur_variant}')
                self.assertIsInstance(it[0], int, msg=f'{cur_variant}')
                self.assertIsInstance(it[1], int, msg=f'{cur_variant}')
                self.assertLess(it[0], it[1], msg=f'{cur_variant}')
                self.assertGreaterEqual(it[0], 0, msg=f'{cur_variant}')
        predicted_variants.sort(key=lambda it: it[1])
        self.assertEqual(predicted_variants[0][1], true_penalty, msg=f'{predicted_variants[0]}')
        self.assertEqual(predicted_variants[0][0], true_entity_bounds, msg=f'{predicted_variants[0]}')

    def test_match_entities_to_tokens_pos03(self):
        tokens = ['a', 'b', 'c', 'd', 'c', 'f', 'g']
        entities = [['b', 'c'], ['c'], ['f']]
        true_entity_bounds = [(1, 3), (4, 5), (5, 6)]
        true_penalty = 0
        predicted_variants = match_entities_to_tokens(tokens, entities, [], 0)
        self.assertIsInstance(predicted_variants, list)
        self.assertGreater(len(predicted_variants), 0)
        for cur_variant in predicted_variants:
            self.assertIsInstance(cur_variant, tuple, msg=f'{cur_variant}')
            self.assertEqual(len(cur_variant), 2, msg=f'{cur_variant}')
            self.assertIsInstance(cur_variant[0], list, msg=f'{cur_variant}')
            self.assertIsInstance(cur_variant[1], int, msg=f'{cur_variant}')
            self.assertGreaterEqual(cur_variant[1], 0, msg=f'{cur_variant}')
            for it in cur_variant[0]:
                self.assertIsInstance(it, tuple, msg=f'{cur_variant}')
                self.assertEqual(len(it), 2, msg=f'{cur_variant}')
                self.assertIsInstance(it[0], int, msg=f'{cur_variant}')
                self.assertIsInstance(it[1], int, msg=f'{cur_variant}')
                self.assertLess(it[0], it[1], msg=f'{cur_variant}')
                self.assertGreaterEqual(it[0], 0, msg=f'{cur_variant}')
        predicted_variants.sort(key=lambda it: it[1])
        self.assertEqual(predicted_variants[0][1], true_penalty, msg=f'{predicted_variants[0]}')
        self.assertEqual(predicted_variants[0][0], true_entity_bounds, msg=f'{predicted_variants[0]}')

    def test_match_entities_to_tokens_pos04(self):
        tokens = ['a', 'b', 'c', 'd', 'c', 'f', 'g']
        entities = [['b', 'c'], ['c'], ['c', 'f']]
        true_entity_bounds = [(1, 3), (4, 5), (4, 6)]
        true_penalty = 0
        predicted_variants = match_entities_to_tokens(tokens, entities, [], 0)
        self.assertIsInstance(predicted_variants, list)
        self.assertGreater(len(predicted_variants), 0)
        for cur_variant in predicted_variants:
            self.assertIsInstance(cur_variant, tuple, msg=f'{cur_variant}')
            self.assertEqual(len(cur_variant), 2, msg=f'{cur_variant}')
            self.assertIsInstance(cur_variant[0], list, msg=f'{cur_variant}')
            self.assertIsInstance(cur_variant[1], int, msg=f'{cur_variant}')
            self.assertGreaterEqual(cur_variant[1], 0, msg=f'{cur_variant}')
            for it in cur_variant[0]:
                self.assertIsInstance(it, tuple, msg=f'{cur_variant}')
                self.assertEqual(len(it), 2, msg=f'{cur_variant}')
                self.assertIsInstance(it[0], int, msg=f'{cur_variant}')
                self.assertIsInstance(it[1], int, msg=f'{cur_variant}')
                self.assertLess(it[0], it[1], msg=f'{cur_variant}')
                self.assertGreaterEqual(it[0], 0, msg=f'{cur_variant}')
        predicted_variants.sort(key=lambda it: it[1])
        self.assertEqual(predicted_variants[0][1], true_penalty, msg=f'{predicted_variants[0]}')
        self.assertEqual(predicted_variants[0][0], true_entity_bounds, msg=f'{predicted_variants[0]}')

    def test_match_entities_to_tokens_neg01(self):
        tokens = ['a', 'b', 'c', 'd', 'c', 'f', 'g']
        entities = []
        predicted_variants = match_entities_to_tokens(tokens, entities, [], 0)
        self.assertIsInstance(predicted_variants, list)
        self.assertEqual(len(predicted_variants), 1)
        true_entity_bounds = []
        true_penalty = 0
        predicted_variants = match_entities_to_tokens(tokens, entities, [], 0)
        self.assertIsInstance(predicted_variants, list)
        self.assertEqual(len(predicted_variants), 1)
        self.assertIsInstance(predicted_variants[0], tuple)
        self.assertEqual(len(predicted_variants[0]), 2)
        self.assertIsInstance(predicted_variants[0][0], list)
        self.assertIsInstance(predicted_variants[0][1], int)
        self.assertEqual(predicted_variants[0][0], true_entity_bounds)
        self.assertEqual(predicted_variants[0][1], true_penalty)

    def test_match_entities_to_tokens_neg02(self):
        tokens = []
        entities = [['b', 'c'], ['c'], ['c', 'f']]
        true_entity_bounds = []
        true_penalty = 3
        predicted_variants = match_entities_to_tokens(tokens, entities, [], 0)
        self.assertIsInstance(predicted_variants, list)
        self.assertEqual(len(predicted_variants), 1)
        self.assertIsInstance(predicted_variants[0], tuple)
        self.assertEqual(len(predicted_variants[0]), 2)
        self.assertIsInstance(predicted_variants[0][0], list)
        self.assertIsInstance(predicted_variants[0][1], int)
        self.assertEqual(predicted_variants[0][0], true_entity_bounds)
        self.assertEqual(predicted_variants[0][1], true_penalty)


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
