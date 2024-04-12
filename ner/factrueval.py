import codecs
import csv
import os
from typing import Dict, List, Tuple


def load_spans(text: str, span_fname: str) -> Dict[int, Tuple[int, int]]:
    res = dict()
    line_idx = 1
    with codecs.open(span_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                err_msg = f'{span_fname}: line {line_idx} is wrong!'
                parts_of_line = prepline.split()
                if len(parts_of_line) < 9:
                    raise ValueError(err_msg)
                try:
                    span_id = int(parts_of_line[0])
                except:
                    span_id = -1
                if span_id < 0:
                    raise ValueError(err_msg)
                if span_id in res:
                    raise ValueError(err_msg + f' The span ID = {span_id} is duplicated!')
                span_text = ' '.join(parts_of_line[8:])
                if len(span_text) == 0:
                    raise ValueError(err_msg + ' The span text is empty!')
                try:
                    start_pos = int(parts_of_line[2])
                except:
                    start_pos = -1
                if start_pos < 0:
                    raise ValueError(err_msg)
                if start_pos >= len(text):
                    raise ValueError(err_msg + f' The span start = {start_pos} is greater then text length!')
                try:
                    span_length = int(parts_of_line[3])
                except:
                    span_length = -1
                if span_length <= 0:
                    raise ValueError(err_msg)
                if (start_pos + span_length) > len(text):
                    err_msg += f' The span end = {start_pos + span_length} is greater then text length!'
                    raise ValueError(err_msg)
                subtext = ' '.join(text[start_pos:(start_pos + span_length)].strip().split())
                if len(subtext) == 0:
                    raise ValueError(err_msg + ' The sub-text is empty!')
                if subtext != span_text:
                    err_msg += f' The text {subtext} does not correspond to the text {span_text}'
                    raise ValueError(err_msg)
                res[span_id] = (start_pos, start_pos + span_length)
            line_idx += 1
            curline = fp.readline()
    return res


def find_entity_by_char_pos(char_pos: int, entities: List[Tuple[int, int]]) -> int:
    if char_pos < 0:
        return -1
    found_idx = -1
    for idx, (start_, end_) in enumerate(entities):
        if (found_idx >= start_) and (found_idx < end_):
            found_idx = idx
            break
    return found_idx


def load_named_entities(spans: Dict[int, Tuple[int, int]], ne_fname: str) -> Dict[str, List[Tuple[int, int]]]:
    res = dict()
    line_idx = 1
    with codecs.open(ne_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                err_msg = f'{ne_fname}: line {line_idx} is wrong!'
                parts_of_line = prepline.split()
                if len(parts_of_line) < 6:
                    raise ValueError(err_msg)
                try:
                    entity_id = int(parts_of_line[0])
                except:
                    entity_id = -1
                if entity_id < 0:
                    raise ValueError(err_msg)
                if entity_id in res:
                    raise ValueError(err_msg + f' The entity ID = {entity_id} is duplicated!')
                entity_type = parts_of_line[1].strip().lower()
                if entity_type not in {'org', 'location', 'person', 'locorg'}:
                    raise ValueError(err_msg + f' The entity type {entity_type} is unknown!')
                if entity_type == 'org':
                    entity_type = 'organization'
                elif entity_type == 'locorg':
                    entity_type = 'location'
                try:
                    comment_col = parts_of_line.index('#')
                except:
                    comment_col = -1
                if comment_col < 0:
                    raise ValueError(err_msg + f' The comment is not found!')
                if comment_col < 3:
                    raise ValueError(err_msg + f' The comment is incorrect!')
                spans_in_entity = set()
                for col_idx in range(2, comment_col):
                    try:
                        span_id = int(parts_of_line[col_idx])
                    except:
                        span_id = -1
                    if span_id < 0:
                        raise ValueError(err_msg + f' The span ID = {parts_of_line[col_idx]} is wrong!')
                    if span_id not in spans:
                        raise ValueError(err_msg + f' The span ID = {span_id} is unknown!')
                    if span_id in spans_in_entity:
                        raise ValueError(err_msg + f' The span ID = {span_id} is duplicated!')
                    spans_in_entity.add(span_id)
                if entity_type not in res:
                    res[entity_type] = []
                sorted_spans_in_entity = sorted(
                    list(spans_in_entity),
                    key=lambda it: (spans[it][0], spans[it][1])
                )
                entity_start = spans[sorted_spans_in_entity[0]][0]
                entity_end = spans[sorted_spans_in_entity[-1]][1]
                res[entity_type].append((entity_start, entity_end))
            line_idx += 1
            curline = fp.readline()
    for entity_type in res:
        res[entity_type].sort(key=lambda it: (it[0], it[1]))
    return res


def load_sample(dataset_path: str, item_name: str,
                join_nested_entities: bool = True) -> Tuple[str, Dict[str, List[Tuple[int, int]]]]:
    if not os.path.isdir(dataset_path):
        raise ValueError(f'The directory "{dataset_path}" does not exist!')
    txt_fname = os.path.join(dataset_path, item_name + '.txt')
    span_fname = os.path.join(dataset_path, item_name + '.spans')
    ne_fname = os.path.join(dataset_path, item_name + '.objects')
    if not os.path.isfile(txt_fname):
        raise ValueError(f'The file "{txt_fname}" does not exist!')
    if not os.path.isfile(span_fname):
        raise ValueError(f'The file "{span_fname}" does not exist!')
    if not os.path.isfile(ne_fname):
        raise ValueError(f'The file "{ne_fname}" does not exist!')
    full_text = ''
    with codecs.open(txt_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        new_line = fp.readline()
        while len(new_line) > 0:
            full_text += new_line.replace('\r', '\n')
            new_line = fp.readline()
    if len(full_text.strip()) == 0:
        raise ValueError(f'The file "{txt_fname}" is empty!')
    full_text = full_text.rstrip()
    all_spans = load_spans(full_text, span_fname)
    entities_in_text = load_named_entities(all_spans, ne_fname)
    return full_text, entities_in_text
