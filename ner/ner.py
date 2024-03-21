from typing import List, Tuple

from nltk import wordpunct_tokenize


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
    tokens_of_text = wordpunct_tokenize(source_text)
    start_pos = 0
    entity_labels = []
    for cur_entity in entities:
        postprocessed_entity = cur_entity.strip()
        while postprocessed_entity.endswith('</s>'):
            postprocessed_entity = postprocessed_entity[:-4].strip()
        tokens_of_entity = wordpunct_tokenize(postprocessed_entity)
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
                err_msg = f'The entity {cur_entity} is not found in the text {source_text}'
                raise ValueError(err_msg)
    if start_pos < len(tokens_of_text):
        for _ in range(len(tokens_of_text) - start_pos):
            entity_labels.append('O')
    if len(entity_labels) != len(tokens_of_text):
        err_msg = (f'The number of tokens does not correspond to the number of entity labels! '
                   f'{len(tokens_of_text)} != {len(entity_labels)}')
        raise ValueError(err_msg)
    return list(zip(tokens_of_text, entity_labels))
