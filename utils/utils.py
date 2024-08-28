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
    target_text = strip_zeros(str(result))
    selected_instruction = random.choice(variants_of_prompt).format(inp=input_text)
    return selected_instruction, target_text
