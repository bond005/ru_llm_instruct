import codecs
from typing import List, Tuple

from nltk import wordpunct_tokenize


def load_samples(fname: str) -> List[Tuple[str, str]]:
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
    
    return texts