import codecs
import copy
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
                try:
                    span_tokens_number = int(parts_of_line[5])
                except:
                    span_tokens_number = 0
                if span_tokens_number < 1:
                    raise ValueError(err_msg)
                span_text = ' '.join(parts_of_line[(7 + span_tokens_number):])
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
                if span_id in res:
                    if res[span_id] != (start_pos, start_pos + span_length):
                        raise ValueError(err_msg + f' The span ID = {span_id} is duplicated!')
                res[span_id] = (start_pos, start_pos + span_length)
            line_idx += 1
            curline = fp.readline()
    return res


def join_nested_entities(entities: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(entities) == 0:
        return []
    if len(entities) == 1:
        return entities
    joined_entities = copy.copy(entities)
    n = len(joined_entities)
    is_changed = False
    idx1 = 0
    while idx1 < n:
        overlapped = []
        entity_start = joined_entities[idx1][0]
        entity_end = joined_entities[idx1][1]
        for idx2 in range(n):
            if idx2 == idx1:
                continue
            if joined_entities[idx2][1] <= entity_start:
                continue
            if entity_end <= joined_entities[idx2][0]:
                continue
            overlapped.append(idx2)
        if len(overlapped) > 0:
            overlapped.sort(key=lambda it: (joined_entities[it][0], joined_entities[it][1]))
            new_entity_start = min(entity_start, joined_entities[overlapped[0]][0])
            new_entity_end = max(entity_end, joined_entities[overlapped[-1]][1])
            removed = [joined_entities[it] for it in overlapped] + [joined_entities[idx1]]
            for val in removed:
                joined_entities.remove(val)
                n -= 1
            joined_entities.append((new_entity_start, new_entity_end))
            idx1 = 0
            del removed
            is_changed = True
        else:
            idx1 += 1
        del overlapped
    joined_entities.sort(key=lambda it: (it[0], it[1]))
    while is_changed:
        n = len(joined_entities)
        is_changed = False
        idx1 = 0
        while idx1 < n:
            overlapped = []
            entity_start = joined_entities[idx1][0]
            entity_end = joined_entities[idx1][1]
            for idx2 in range(n):
                if idx2 == idx1:
                    continue
                if joined_entities[idx2][1] <= entity_start:
                    continue
                if entity_end <= joined_entities[idx2][0]:
                    continue
                overlapped.append(idx2)
            if len(overlapped) > 0:
                overlapped.sort(key=lambda it: (joined_entities[it][0], joined_entities[it][1]))
                new_entity_start = min(entity_start, joined_entities[overlapped[0]][0])
                new_entity_end = max(entity_end, joined_entities[overlapped[-1]][1])
                removed = [joined_entities[it] for it in overlapped] + [joined_entities[idx1]]
                for val in removed:
                    joined_entities.remove(val)
                    n -= 1
                joined_entities.append((new_entity_start, new_entity_end))
                idx1 = 0
                del removed
                is_changed = True
            else:
                idx1 += 1
            del overlapped
        joined_entities.sort(key=lambda it: (it[0], it[1]))
    return joined_entities


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
                if len(parts_of_line) < 5:
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
                entity_start = min([spans[i][0] for i in spans_in_entity])
                entity_end = max([spans[i][1] for i in spans_in_entity])
                res[entity_type].append((entity_start, entity_end))
                del spans_in_entity
            line_idx += 1
            curline = fp.readline()
    for entity_type in res:
        res[entity_type].sort(key=lambda it: (it[0], it[1]))
    return res


def extend_entity_bounds(text: str, entity_start: int, entity_end: int) -> Tuple[int, int]:
    is_changed = False
    if entity_start > 0:
        if text[entity_start - 1] == '«':
            entity_start_ = entity_start - 1
            is_changed = True
        else:
            entity_start_ = entity_start
    else:
        entity_start_ = entity_start
    if entity_end < len(text):
        if text[entity_end] == '»':
            entity_end_ = entity_end + 1
            is_changed = True
        else:
            entity_end_ = entity_end
    else:
        entity_end_ = entity_end
    if not is_changed:
        return entity_start, entity_end
    n1 = 0
    start_pos = entity_start_
    found_idx = text[start_pos:entity_end_].find('«')
    while found_idx >= 0:
        n1 += 1
        start_pos += (found_idx + 1)
        found_idx = text[start_pos:entity_end_].find('«')
    n2 = 0
    start_pos = entity_start_
    found_idx = text[start_pos:entity_end_].find('»')
    while found_idx >= 0:
        n2 += 1
        start_pos += (found_idx + 1)
        found_idx = text[start_pos:entity_end_].find('»')
    if n1 == n2:
        return entity_start_, entity_end_
    return entity_start, entity_end


def load_sample(dataset_path: str, item_name: str) -> Tuple[str, Dict[str, List[Tuple[int, int]]]]:
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
    with open(txt_fname, 'rb') as fp:
        full_text = fp.read().decode('utf-8').replace('\r', '')
    if len(full_text.strip()) == 0:
        raise ValueError(f'The file "{txt_fname}" is empty!')
    full_text = full_text.rstrip()
    all_spans = load_spans(full_text, span_fname)
    entities_in_text = load_named_entities(all_spans, ne_fname)
    for k in sorted(list(entities_in_text.keys())):
        entities_in_text[k] = join_nested_entities([extend_entity_bounds(full_text, *it) for it in entities_in_text[k]])
    return full_text, entities_in_text
