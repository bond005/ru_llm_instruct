import codecs
from typing import List, Tuple

from nltk import wordpunct_tokenize


def load_samples_from_taiga(fname: str) -> List[Tuple[str, str]]:
    paragraphs = []
    new_paragraph = ''
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = ' '.join((curline.strip().replace('--', '—').split()))
            if len(prepline) > 0:
                words_in_line = list(filter(lambda it: it.isalpha(), wordpunct_tokenize(prepline)))
                if len(words_in_line) < 1:
                    if len(new_paragraph) > 0:
                        paragraphs.append(' '.join(new_paragraph.split()))
                        new_paragraph = ''
                    paragraphs.append(prepline)
                else:
                    if len(new_paragraph) == 0:
                        new_paragraph = prepline
                    else:
                        if new_paragraph[-1] in {'.', '?', '!', '…'}:
                            paragraphs.append(' '.join(new_paragraph.split()))
                            new_paragraph = prepline
                        else:
                            if prepline.startswith('—') or prepline.startswith('-') or prepline[0].isdigit():
                                paragraphs.append(' '.join(new_paragraph.split()))
                                new_paragraph = prepline
                            else:
                                if words_in_line[0].istitle() or ' '.join(words_in_line).isupper():
                                    paragraphs.append(' '.join(new_paragraph.split()))
                                    new_paragraph = prepline
                                else:
                                    new_paragraph += ' ' + prepline
            else:
                if len(new_paragraph) > 0:
                    paragraphs.append(' '.join(new_paragraph.split()))
                    new_paragraph = ''
            curline = fp.readline()
    if len(new_paragraph) > 0:
        paragraphs.append(' '.join(new_paragraph.split()))
    if len(paragraphs) == 0:
        print(f'There are no paragraphs in the "{fname}".')
        return []
    texts = []
    new_text = ''
    for cur_paragraph in paragraphs:
        words_in_paragraph = list(filter(lambda it: it.isalpha(), wordpunct_tokenize(cur_paragraph)))
        if len(words_in_paragraph) < 1:
            if len(new_text) > 0:
                texts.append(new_text.strip())
                new_text = ''
        else:
            if len(new_text) == 0:
                new_text = cur_paragraph
            else:
                new_text += ('\n' + cur_paragraph)
    if len(new_text) > 0:
        texts.append(new_text.strip())
    if len(texts) < 1:
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
