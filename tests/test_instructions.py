import os
import random
import sys
import unittest

try:
    from instructions.instructions import get_task_type
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from instructions.instructions import get_task_type


class TestInstructions(unittest.TestCase):
    def test_get_task_type_pos01(self):
        src = ('<LM>Найди, пожалуйста, все именованные сущности типа "Местоположение" в следующем тексте и '
               'выпиши список таких сущностей. Президент США Барак Обама в пятницу , 19 июня , назвал кандидатов '
               'на посты послов в Грузии и Таджикистане , сообщила пресс - служба Белого дома .')
        true_task_id = 6
        self.assertEqual(true_task_id, get_task_type(src, True))

    def test_get_task_type_pos02(self):
        src = ('Найди, пожалуйста, все именованные сущности типа "Местоположение" в следующем тексте и '
               'выпиши список таких сущностей. Президент США Барак Обама в пятницу , 19 июня , назвал кандидатов '
               'на посты послов в Грузии и Таджикистане , сообщила пресс - служба Белого дома .')
        true_task_id = 6
        self.assertEqual(true_task_id, get_task_type(src, False))

    def test_get_task_type_pos03(self):
        src = 'Как вы думаете, когда мы в следующий раз отправим человека на Луну?'
        true_task_id = -1
        self.assertEqual(true_task_id, get_task_type(src, False))

    def test_get_task_type_neg01(self):
        src = ('Президент США Барак Обама. Найди, пожалуйста, все именованные сущности типа "Местоположение" в '
               'следующем тексте и выпиши список таких сущностей. Президент США Барак Обама в пятницу , 19 июня , '
               'назвал кандидатов на посты послов в Грузии и Таджикистане , сообщила пресс - служба Белого дома .')
        with self.assertRaises(ValueError):
            _ = get_task_type(src, False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
