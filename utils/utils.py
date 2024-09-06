import random
from typing import List, Tuple, Union

from nltk import wordpunct_tokenize
import numpy as np
from spacy import Language


def tokenize_text(s: str) -> List[str]:
    words_ = wordpunct_tokenize(s)
    words = []
    for cur in words_:
        if cur.isalnum():
            words.append(cur)
        else:
            words += list(cur)
    return words


def levenshtein(seq1: List[str], seq2: List[str]) -> float:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    int(matrix[x - 1, y]) + 1,
                    int(matrix[x - 1, y - 1]),
                    int(matrix[x, y - 1]) + 1
                )
            else:
                matrix[x, y] = min(
                    int(matrix[x - 1, y]) + 1,
                    int(matrix[x - 1, y - 1]) + 1,
                    int(matrix[x, y - 1]) + 1
                )
    return float(matrix[size_x - 1, size_y - 1])


def calculate_word_error_rate(predicted: List[str], reference: List[str]) -> Tuple[float, float]:
    dist = levenshtein(predicted, reference)
    return dist, float(len(reference))


def process_multiline(s: str) -> Union[str, List[str]]:
    lines = list(filter(
        lambda it2: len(it2) > 0,
        map(
            lambda it1: ' '.join(it1.strip().split()).strip(),
            s.split('\n')
        )
    ))
    if len(lines) > 1:
        return lines
    return ' '.join(s.split())


def process_target(s: str) -> str:
    s_ = s.strip()
    while s_.endswith('</s>'):
        s_ = s_[:-4].strip()
    return s_


def is_punctuation(s: str) -> bool:
    characters = list(set(s))
    if len(characters) == 0:
        return False
    ok = True
    for c in characters:
        if c.isalnum():
            ok = False
            break
    return ok


def normalize_text(s: str, spacy_nlp: Language) -> str:
    doc = spacy_nlp(s)
    normalized = ' '.join(filter(
        lambda it2: (not is_punctuation(it2)) and (len(it2) > 0),
        map(lambda it1: it1.lemma_.lower(), doc)
    )).strip()
    del doc
    return normalized


def split_text_by_sentences(text: str, nlp: Language) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    doc = nlp(text)
    tokens = [(it.idx, it.idx + len(it.text)) for it in doc]
    sentences = list(doc.sents)
    sentence_bounds = []
    for cur_sent in sentences:
        sent_start = cur_sent.start
        sent_end = cur_sent.end
        sentence_bounds.append((sent_start, sent_end))
    del sentences, doc
    return sentence_bounds, tokens


def split_long_text(long_text: str, maxlen: int, nlp: Language) -> List[Tuple[int, int]]:
    if len(long_text) <= maxlen:
        shorter_texts = [(0, len(long_text))]
    else:
        sentences, tokens = split_text_by_sentences(long_text, nlp)
        if len(sentences) > 1:
            middle_sentence_idx = (len(sentences) - 1) // 2
            middle_token_idx = sentences[middle_sentence_idx][1] - 1
            del middle_sentence_idx
        else:
            middle_token_idx = (len(tokens) - 1) // 2
        middle_char_idx = tokens[middle_token_idx][1]
        del sentences, tokens, middle_token_idx
        shorter_texts = split_long_text(long_text[:middle_char_idx], maxlen, nlp)
        shorter_texts += list(map(lambda it: (it[0] + middle_char_idx, it[1] + middle_char_idx),
                                  split_long_text(long_text[middle_char_idx:], maxlen, nlp)))
    return shorter_texts


def strip_zeros(s: str) -> str:
    if (not s.endswith('0')) or (len(s) < 2):
        return s
    s_ = s[:-1]
    while len(s_) > 1:
        if not s_.endswith('0'):
            break
        s_ = s_[:-1]
    if (s_[-1] == '.') or (s_[-1] == ','):
        s_ = s_[:-1]
    return s_


def generate_arithmetic_sample() -> Tuple[str, str]:
    variants_of_prompt = [
        'Посчитай, пожалуйста, сколько будет {inp}?',
        '{inp}=',
        '{inp} = ',
        'Представь себя опытным специалистом по арифметике и посчитай, чему равно {inp}:',
        'Вычисли {inp}',
        'Вычисли {inp}=',
        'Вычисли {inp} = ',
        'Рассчитай {inp}=',
        'Подсчитай {inp}=',
        'Сколько будет {inp}?',
        'Сколько будет {inp} =',
        'Сколько будет {inp}=',
        'Сколько будет {inp} ?'
    ]
    arithmetic_operation = random.choice(['+', '-', '*', '/'])
    if random.random() > 0.5:
        first_item = round(200 * (random.random() - 0.5), 6)
        second_item = round(200 * (random.random() - 0.5), 6)
    else:
        first_item = random.randint(-100, 100)
        second_item = random.randint(-100, 100)
    if arithmetic_operation == '+':
        result = first_item + second_item
    elif arithmetic_operation == '-':
        result = first_item - second_item
    elif arithmetic_operation == '*':
        result = first_item * second_item
    else:
        if abs(second_item) < 1e-6:
            arithmetic_operation = '*'
            result = first_item * second_item
        else:
            result = first_item / second_item
    result = round(result, 6)
    input_text = strip_zeros(str(first_item))
    if random.random() > 0.5:
        input_text += ' '
    input_text += arithmetic_operation
    if random.random() > 0.5:
        input_text += ' '
    input_text += strip_zeros(str(second_item))
    target_text = strip_zeros(str(result)) + '</s>'
    selected_instruction = random.choice(variants_of_prompt).format(inp=input_text)
    return selected_instruction, target_text


def generate_sample_with_comparison() -> Tuple[str, str]:
    variants_of_promt = [
        'Какое из чисел {op}: {a} или {b}?',
        'Ответь, какое из двух чисел {op}: {a} или же {b}?',
        'Как ты думаешь, какое из двух чисел {op}: {a} или, может быть, {b}?',
        'Подскажи, какое из чисел {op}: {a} или же {b}?',
        'Подскажи, пожалуйста, какое из двух чисел {op}: {a} или же {b}?',
        'Как ты считаешь, какое из чисел {op}: {a} или, может быть, {b}?'
    ]
    operation = random.choice(['больше', 'меньше', 'min', 'max', 'minimum', 'maximum'])
    first_value = random.randint(-200, 200)
    second_value = random.randint(-200, 200)
    while second_value == first_value:
        second_value = random.randint(-200, 200)
    if operation in {'больше', 'max', 'maximum'}:
        target_text = str(max(first_value, second_value))
    else:
        target_text = str(min(first_value, second_value))
    if operation in {'больше', 'меньше'}:
        input_text = random.choice(variants_of_promt).format(op=operation, a=first_value, b=second_value)
    else:
        input_text = operation
        if random.random() > 0.5:
            input_text += ' '
        input_text += '('
        if random.random() > 0.5:
            input_text += ' '
        input_text += str(first_value)
        if random.random() > 0.5:
            input_text += ' '
        input_text += ','
        if random.random() > 0.5:
            input_text += ' '
        input_text += str(second_value)
        if random.random() > 0.5:
            input_text += ' '
        input_text += ')'
        if random.random() > 0.5:
            input_text += ' '
        if random.random() > 0.3:
            if random.random() > 0.5:
                input_text += '='
                if random.random() > 0.5:
                    input_text += ' '
                if random.random() > 0.5:
                    input_text += '?'
            else:
                input_text += '?'
                if random.random() > 0.5:
                    input_text += ' '
    return input_text, target_text + '</s>'


def generate_sample_with_choice() -> Tuple[str, str]:
    variants_of_prompt = [
        'Дан массив чисел {arr}. Подскажи, пожалуйста, какое число в этом массиве самое {op}?',
        'У тебя есть список чисел {arr}. Определи, какое число в этом списке самое {op}?',
        'Определи в массиве {arr}, какое из чисел самое {op}?',
        'Определи в списке чисел {arr}, какое из них самое {op}?',
        'Посмотри на набор чисел {arr} и выясни, какое из них самое {op}?',
        'Проанализируй выборку чисел {arr}. Какое из этих чисел самое {op}?',
        'Есть список чисел {arr}. Какое число в этом списке самое {op}?',
        'У тебя есть массив чисел {arr}. Определи, какое число в этом массиве самое {op}?',
        'Определи в последовательности {arr}, какое из приведённых чисел самое {op}?',
        'Определи в наборе чисел {arr}, какое самое {op}?',
        'Посмотри на множество чисел {arr} и выясни, какое из этих чисел самое {op}?',
        'Проанализируй множество чисел {arr}. Какое из данных чисел самое {op}?'
    ]
    variants_of_question_2_4 = [
        'У тебя есть {n} варианта ответа. Какой из них - правильный? Запиши только букву верного варианта: {vars}.',
        'Перечислено {n} варианта ответа. Какой из вариантов является правильным? Запиши только одну букву {vars}.',
        'Из {n} возможных вариантов ответа выбери правильный и запиши {vars}.'
    ]
    variants_of_question_5_n = [
        'У тебя есть {n} вариантов ответа. Какой из них - правильный? Запиши только букву верного варианта: {vars}.',
        'Перечислено {n} вариантов ответа. Какой из вариантов является правильным? Запиши только одну букву {vars}.',
        'Из {n} возможных вариантов ответа выбери правильный и запиши {vars}.'
    ]
    array_size = random.randint(5, 20)
    integer_array = [random.randint(-200, 200) for _ in range(array_size)]
    if random.random() > 0.5:
        comparison = 'большое'
        true_value = max(integer_array)
    else:
        comparison = 'маленькое'
        true_value = min(integer_array)
    if random.random() > 0.5:
        array_as_string = ','.join([str(x) for x in integer_array])
    else:
        array_as_string = ', '.join([str(x) for x in integer_array])
    if random.random() > 0.5:
        array_as_string_ = '['
        if random.random() > 0.5:
            array_as_string_ += ' '
        array_as_string_ += array_as_string
        if random.random() > 0.5:
            array_as_string_ += ' '
        array_as_string_ += ']'
        array_as_string = array_as_string_
        del array_as_string_
    input_text = random.choice(variants_of_prompt).format(arr=array_as_string, op=comparison)
    if random.random() > 0.5:
        input_text += ' '
    else:
        input_text += '\n'
    number_of_variants = random.randint(2, min(6, array_size - 2))
    if random.random() > 0.5:
        full_identifiers_of_choice = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                      'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        is_choice_letter = True
    else:
        full_identifiers_of_choice = [str(val) for val in range(1, 101)]
        is_choice_letter = False

    identifiers_of_choice = random.sample(population=full_identifiers_of_choice, k=number_of_variants)
    if random.random() > 0.5:
        identifiers_as_string = ','.join(identifiers_of_choice[:(number_of_variants - 1)])
    else:
        identifiers_as_string = ', '.join(identifiers_of_choice[:(number_of_variants - 1)])
    identifiers_as_string += ' или ' + identifiers_of_choice[number_of_variants - 1]
    if is_choice_letter:
        identifiers_as_string = 'одной буквой: ' + identifiers_as_string
    else:
        identifiers_as_string = 'одним числом: ' + identifiers_as_string
    if number_of_variants > 4:
        input_text += random.choice(variants_of_question_5_n).format(n=number_of_variants, vars=identifiers_as_string)
    else:
        input_text += random.choice(variants_of_question_2_4).format(n=number_of_variants, vars=identifiers_as_string)
    if random.random() > 0.5:
        input_text += ' Варианты ответа:'
    else:
        input_text += '\nВарианты ответа:'
    variants = random.sample(population=integer_array, k=number_of_variants)
    try:
        true_idx = variants.index(true_value)
    except:
        true_idx = -1
    if (random.random() > 0.3) or (number_of_variants < 3):
        if true_idx < 0:
            variants.remove(random.choice(variants))
            variants.append(true_value)
            random.shuffle(variants)
            true_idx = variants.index(true_value)
        input_text += '\n' + '\n'.join([f'{identifiers_of_choice[idx]}. {val}' for idx, val in enumerate(variants)])
        target_text = identifiers_of_choice[true_idx]
    else:
        while true_idx >= 0:
            variants.remove(true_value)
            variants.append(random.choice(integer_array))
            try:
                true_idx = variants.index(true_value)
            except:
                true_idx = -1
        true_idx = random.randint(0, number_of_variants - 1)
        input_text += '\n' + '\n'.join(
            [f'{identifiers_of_choice[idx]}. {val if (idx != true_idx) else "здесь нет правильного ответа"}'
             for idx, val in enumerate(variants)]
        )
        target_text = identifiers_of_choice[true_idx]
    if random.random() > 0.5:
        input_text += '\n'
        if random.random() > 0.5:
            input_text += 'Ответ:'
        else:
            input_text += 'Твой ответ:'
        if random.random() > 0.5:
            input_text += ' '
    return input_text, target_text + '</s>'
