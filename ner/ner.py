from typing import List, Tuple

from utils.utils import tokenize_text


def find_subphrase(full_phrase: List[str], subphrase: List[str]) -> int:
    n = len(subphrase)
    subphrase_ = ' '.join(subphrase).lower()
    if n > len(full_phrase):
        return -1
    if n == len(full_phrase):
        if subphrase_ == ' '.join(full_phrase).lower():
            return 0
        return -1
    found_idx = -1
    for idx in range(len(full_phrase) - n + 1):
        if subphrase_ == ' '.join(full_phrase[idx:(idx + n)]).lower():
            found_idx = idx
            break
    return found_idx


def find_entities_in_text(source_text: str, entities: List[str], entity_class: str,
                          raise_exception: bool = False) -> List[Tuple[str, str]]:
    tokens_of_text = tokenize_text(source_text)
    start_pos = 0
    entity_labels = []
    for cur_entity in entities:
        postprocessed_entity = cur_entity.strip()
        while postprocessed_entity.endswith('</s>'):
            postprocessed_entity = postprocessed_entity[:-4].strip()
        tokens_of_entity = tokenize_text(postprocessed_entity)
        found_token_idx = find_subphrase(tokens_of_text[start_pos:], tokens_of_entity)
        if found_token_idx >= 0:
            found_token_idx += start_pos
            if found_token_idx - start_pos > 0:
                for _ in range(found_token_idx - start_pos):
                    entity_labels.append('O')
            entity_labels.append(f'B-{entity_class}')
            if len(tokens_of_entity) > 1:
                for _ in range(len(tokens_of_entity) - 1):
                    entity_labels.append(f'I-{entity_class}')
            start_pos = found_token_idx + len(tokens_of_entity)
        else:
            if raise_exception:
                err_msg = f'The entity {cur_entity} from entity list {entities} is not found in the text {source_text}'
                raise ValueError(err_msg)
    if start_pos < len(tokens_of_text):
        for _ in range(len(tokens_of_text) - start_pos):
            entity_labels.append('O')
    if len(entity_labels) != len(tokens_of_text):
        err_msg = (f'The number of tokens does not correspond to the number of entity labels! '
                   f'{len(tokens_of_text)} != {len(entity_labels)}')
        raise ValueError(err_msg)
    return list(zip(tokens_of_text, entity_labels))


def match_entities_to_tokens(tokens: List[str], prepared_entities: List[List[str]],
                             entity_bounds: List[Tuple[int, int]],
                             penalty: int, start_pos: int = 0) -> List[Tuple[List[Tuple[int, int]], int]]:
    if len(prepared_entities) == 0:
        return [([], 0)]
    if (len(tokens) == 0) and (len(prepared_entities) > 0):
        return [(entity_bounds, penalty + len(prepared_entities))]
    new_penalty = penalty
    found_token_idx = find_subphrase(tokens, prepared_entities[0])
    if found_token_idx >= 0:
        entity_bounds_ = entity_bounds + [(
            found_token_idx + start_pos,
            found_token_idx + len(prepared_entities[0]) + start_pos
        )]
        if len(prepared_entities) > 1:
            res = match_entities_to_tokens(tokens[(found_token_idx + len(prepared_entities[0])):],
                                           prepared_entities[1:], entity_bounds_,
                                           new_penalty, found_token_idx + len(prepared_entities[0]) + start_pos)
            res += match_entities_to_tokens(tokens[found_token_idx:], prepared_entities[1:], entity_bounds_,
                                            new_penalty, found_token_idx + start_pos)
        else:
            res = [(entity_bounds_, new_penalty)]
    else:
        new_penalty += 1
        if len(prepared_entities) > 1:
            res = match_entities_to_tokens(tokens, prepared_entities[1:], entity_bounds, new_penalty, start_pos)
        else:
            res = [(entity_bounds, new_penalty)]
    return res


def check_nested(bounds: List[Tuple[int, int]]) -> bool:
    if len(bounds) < 2:
        return False
    ok = False
    for idx in range(1, len(bounds)):
        if bounds[idx - 1][1] > bounds[idx][0]:
            ok = True
            break
    return ok


def calculate_entity_bounds(source_text: str, entities: List[str], nested: bool = False) -> List[Tuple[int, int]]:
    tokens_of_text = tokenize_text(source_text)
    if ' '.join(tokens_of_text).lower() == 'в этом тексте нет именованных сущностей такого типа':
        return []
    token_bounds = []
    start_pos = 0
    for cur_token in tokens_of_text:
        found_char_idx = source_text[start_pos:].find(cur_token)
        if found_char_idx < 0:
            err_msg = f'The token {cur_token} is not found in the text {source_text}'
            raise RuntimeError(err_msg)
        token_bounds.append((found_char_idx + start_pos, found_char_idx + start_pos + len(cur_token)))
        start_pos = found_char_idx + start_pos + len(cur_token)
    prepared_entities = []
    for cur_entity in entities:
        postprocessed_entity = cur_entity.strip()
        while postprocessed_entity.endswith('</s>'):
            postprocessed_entity = postprocessed_entity[:-4].strip()
        prepared_entities.append(tokenize_text(postprocessed_entity))
    variants_of_entity_bounds = match_entities_to_tokens(tokens_of_text, prepared_entities, [], 0)
    variants_of_entity_bounds.sort(key=lambda it: (it[1], abs(len(entities) - len(it[0]))))
    variants_of_entity_bounds = list(filter(lambda it: len(it[0]) > 0, variants_of_entity_bounds))
    variants_of_entity_bounds_ = [
        (
            sorted(list(set(cur[0])), key=lambda x: (x[0], x[1])),
            cur[1]
        )
        for cur in variants_of_entity_bounds
    ]
    del variants_of_entity_bounds
    entity_bounds = []
    if len(variants_of_entity_bounds_) > 0:
        if not nested:
            variants_of_entity_bounds_ = list(filter(lambda it: not check_nested(it[0]), variants_of_entity_bounds_))
        for entity_token_start, entity_token_end in variants_of_entity_bounds_[0][0]:
            entity_bounds.append((
                token_bounds[entity_token_start][0],
                token_bounds[entity_token_end - 1][1]
            ))
    return entity_bounds
