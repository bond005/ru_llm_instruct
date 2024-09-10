import codecs
import os
from typing import Dict, List, Tuple
import warnings

from nltk import wordpunct_tokenize
import spacy


def split_text_into_paragraphs(text: str) -> List[Tuple[int, int]]:
    paragraphs = list(filter(lambda x2: len(x2) > 0, map(lambda x1: x1.strip(), text.split('\n'))))
    if len(paragraphs) == 0:
        return []
    start_pos = 0
    bounds_of_paragraphs = []
    for cur_paragraph in paragraphs:
        found_idx = text[start_pos:].find(cur_paragraph)
        if found_idx < 0:
            raise ValueError(f'The text cannot be split into paragraphs! {text}')
        bounds_of_paragraphs.append((found_idx + start_pos, found_idx + start_pos + len(cur_paragraph)))
        start_pos = found_idx + start_pos + len(cur_paragraph)
    return bounds_of_paragraphs


def select_entitites_in_paragraph(full_entities: Dict[str, List[Tuple[int, int]]],
                                  paragraph_start: int, paragraph_end: int) -> Dict[str, List[Tuple[int, int]]]:
    selected = dict()
    for ne_class in full_entities:
        for ne_start, ne_end in full_entities[ne_class]:
            if (ne_start >= paragraph_start) and (ne_end <= paragraph_end):
                if ne_class not in selected:
                    selected[ne_class] = []
                selected[ne_class].append((ne_start - paragraph_start, ne_end - paragraph_start))
            else:
                if (ne_start <= paragraph_start) and (ne_end >= paragraph_end):
                    ok = False
                elif (ne_start <= paragraph_start) and (ne_end > paragraph_start):
                    ok = False
                elif (ne_start < paragraph_end) and (ne_end > paragraph_end):
                    ok = False
                else:
                    ok = True
                if not ok:
                    err_msg = (f'The entity ({ne_class}, {ne_start}, {ne_end}) does not correspond '
                               f'to the paragraph ({paragraph_start}, {paragraph_end}).')
                    raise ValueError(err_msg)
    return selected


def find_entity_for_token(text: str, token_start: int, token_end: int,
                          entities: Dict[str, List[Tuple[int, int]]]) -> List[Tuple[str, int]]:
    found_entities = []
    for ne_class in sorted(list(entities.keys())):
        for ne_idx, (ne_start, ne_end) in enumerate(entities[ne_class]):
            if (token_start >= ne_start) and (token_end <= ne_end):
                found_entities.append((ne_class, ne_idx))
            else:
                if (token_start <= ne_start) and (token_end >= ne_end):
                    ok = False
                elif (token_start <= ne_start) and (token_end > ne_start):
                    ok = False
                elif (token_start < ne_end) and (token_end > ne_end):
                    ok = False
                else:
                    ok = True
                if not ok:
                    err_msg = (f'The token ({token_start}, {token_end}) {text[token_start:token_end]} '
                               f'does not correspond to the named entity ({ne_class}, {ne_start}, {ne_end}) '
                               f'{text[ne_start:ne_end]}.')
                    raise ValueError(err_msg)
    if len(found_entities) > 0:
        found_entities.sort(key=lambda it: (entities[it[0]][it[1]][0], it[0], entities[it[0]][it[1]][1]))
    return found_entities


def split_into_subtokens(token_text: str) -> List[Tuple[int, int]]:
    subtokens_ = wordpunct_tokenize(token_text)
    subtokens = []
    for it in subtokens_:
        if it.isalnum():
            subtokens.append(it)
        else:
            subtokens += list(it)
    del subtokens_
    bounds_of_subtokens = []
    start_pos = 0
    for it in subtokens:
        found_idx = token_text[start_pos:].find(it)
        if found_idx < 0:
            raise ValueError(f'The token {token_text} cannot be splitted!')
        bounds_of_subtokens.append((found_idx + start_pos, found_idx + start_pos + len(it)))
        start_pos = found_idx + start_pos + len(it)
    return bounds_of_subtokens


def load_sample(fname: str) -> List[Tuple[str, List[Tuple[str, str, str, str]]]]:
    base_fname = os.path.basename(fname)
    point_pos = base_fname.rfind('.')
    if point_pos < 0:
        fname_without_extension = base_fname
    else:
        fname_without_extension = base_fname[:point_pos]
        if base_fname[(point_pos + 1):] not in {'ann', 'txt'}:
            raise IOError(f'The file "{fname}" has unknown format!')
    fname_without_extension = os.path.join(os.path.dirname(fname), fname_without_extension)
    text_fname = fname_without_extension + '.txt'
    annotation_fname = fname_without_extension + '.ann'
    if not os.path.isfile(text_fname):
        raise IOError(f'The file "{text_fname}" does not exist!')
    if not os.path.isfile(annotation_fname):
        raise IOError(f'The file "{annotation_fname}" does not exist!')

    with codecs.open(text_fname, mode='r', encoding='utf-8') as fp:
        full_text = fp.read()

    named_entity_labels: Dict[str, List[Tuple[int, int]]] = dict()
    identifiers = set()
    line_idx = 1
    with codecs.open(annotation_fname, mode='r', encoding='utf-8') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = f'The file "{annotation_fname}": line {line_idx} is wrong!'
                parts_of_line = list(filter(lambda x2: len(x2) > 0, map(lambda x1: x1.strip(), prep_line.split('\t'))))
                if len(parts_of_line) != 3:
                    warnings.warn(err_msg + f' Expected field number is 3, got {len(parts_of_line)}.')
                else:
                    if not parts_of_line[0][1:].isdigit():
                        raise IOError(err_msg)
                    if parts_of_line[0][0] == 'T':
                        new_identifier = int(parts_of_line[0][1:])
                        if new_identifier in identifiers:
                            raise IOError(err_msg + f' Identifier = {new_identifier} is duplicated!')
                        identifiers.add(new_identifier)
                        ne_text = parts_of_line[2].strip()
                        if len(ne_text) == 0:
                            raise IOError(err_msg)
                        ne_info = parts_of_line[1].split()
                        if len(ne_info) != 3:
                            warnings.warn(err_msg)
                        else:
                            if (not ne_info[1].isdigit()) or (not ne_info[2].isdigit()):
                                raise IOError(err_msg)
                            ne_class = ne_info[0]
                            ne_start = int(ne_info[1])
                            ne_end = int(ne_info[2])
                            if ne_end <= ne_start:
                                raise IOError(err_msg)
                            if ne_start < 0:
                                raise IOError(err_msg)
                            if ne_end > len(full_text):
                                raise IOError(err_msg)
                            if full_text[ne_start:ne_end] != ne_text:
                                raise IOError(err_msg + f' {ne_text} != {full_text[ne_start:ne_end]}')
                            if ne_class not in named_entity_labels:
                                named_entity_labels[ne_class] = []
                            named_entity_labels[ne_class].append((ne_start, ne_end))
            cur_line = fp.readline()
            line_idx += 1
    if len(named_entity_labels) == 0:
        raise IOError(f'The file {annotation_fname} is empty!')
    for ne_class in sorted(list(named_entity_labels.keys())):
        named_entity_labels[ne_class] = sorted(named_entity_labels[ne_class], key=lambda it: (it[0], it[1]))

    res = []
    nlp = spacy.load('ru_core_news_sm')
    for paragraph_start, paragraph_end in split_text_into_paragraphs(full_text):
        new_conll2003_sample = []
        cur_text = full_text[paragraph_start:paragraph_end]
        try:
            named_entity_labels_in_paragraph = select_entitites_in_paragraph(named_entity_labels,
                                                                             paragraph_start, paragraph_end)
        except Exception as err:
            raise ValueError(f'The file "{annotation_fname}" contains a wrong data! ' + str(err))
        cur_doc = nlp(cur_text)
        for cur_token in cur_doc:
            pos_tag = cur_token.pos_
            dep_tag = cur_token.dep_
            colon_pos = dep_tag.find(':')
            if colon_pos >= 0:
                dep_tag = dep_tag[:colon_pos].strip()
            if len(dep_tag) == 0:
                err_msg = f'The file "{annotation_fname}" contains a wrong data! The text cannot be parsed. {cur_text}'
                raise ValueError(err_msg)
            pos_tag_bio = 'B-' + pos_tag
            dep_tag_bio = 'B-' + dep_tag
            subtoken_bounds = list(map(
                lambda x: (x[0] + cur_token.idx, x[1] + cur_token.idx),
                split_into_subtokens(cur_token.text)
            ))
            new_conll2003_sample_ = []
            ok = True
            for subtoken_start, subtoken_end in subtoken_bounds:
                try:
                    entities_in_token = find_entity_for_token(cur_text, subtoken_start, subtoken_end,
                                                              named_entity_labels_in_paragraph)
                except Exception as err:
                    warnings.warn(f'The file "{annotation_fname}" contains a wrong data! ' + str(err))
                    entities_in_token = None
                    ok = False
                if not ok:
                    break
                if len(entities_in_token) == 0:
                    ne_tag_bio = 'O'
                else:
                    ne_tag_bio = ''
                    for ne_class, ne_idx in entities_in_token:
                        ne_start = named_entity_labels_in_paragraph[ne_class][ne_idx][0]
                        if subtoken_start == ne_start:
                            ne_tag_bio += ',B-' + ne_class
                        else:
                            ne_tag_bio += ',I-' + ne_class
                    if ne_tag_bio.startswith(','):
                        ne_tag_bio = ne_tag_bio[1:]
                new_conll2003_sample_.append((
                    cur_text[subtoken_start:subtoken_end],
                    pos_tag_bio, dep_tag_bio, ne_tag_bio
                ))
                pos_tag_bio = 'I-' + pos_tag
                dep_tag_bio = 'I-' + dep_tag
            if not ok:
                new_conll2003_sample.clear()
                break
            new_conll2003_sample += new_conll2003_sample_
            del new_conll2003_sample_
        del named_entity_labels_in_paragraph, cur_doc
        if len(new_conll2003_sample) > 0:
            res.append((cur_text, new_conll2003_sample))
        del new_conll2003_sample
    return res
