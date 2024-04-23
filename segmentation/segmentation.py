import codecs
from typing import List, Tuple

from nltk import wordpunct_tokenize


def load_samples_from_taiga(fname: str) -> List[Tuple[str, str]]:
    texts = []
    new_text = []
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                words_in_line = wordpunct_tokenize(prepline)
                if len(words_in_line) < 5:
                    if len(new_text) > 0:
                        texts.append('\n'.join(new_text))
                    del new_text
                    new_text = []
                else:
                    new_text.append(' '.join(prepline.split()))
            curline = fp.readline()
    if len(new_text) > 0:
        texts.append('\n'.join(new_text))
    if len(texts) == 0:
        print(f'There are no long texts in the "{fname}".')
        return []
    text_pairs = []
    n_multiparagraph_samples = 0
    for cur_text in texts:
        paragraphs = cur_text.split('\n')
        if len(paragraphs) > 1:
            n_multiparagraph_samples += 1
            text_pairs.append((' '.join(paragraphs), cur_text))
        else:
            text_pairs.append((cur_text, cur_text))
    print(f'There are {n_multiparagraph_samples} multiparagraph texts from {len(texts)} in the "{fname}".')
    return text_pairs
