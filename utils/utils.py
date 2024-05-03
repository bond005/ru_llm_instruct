from typing import List, Tuple, Union

import numpy as np
from spacy import Language


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
