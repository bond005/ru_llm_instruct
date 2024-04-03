import codecs
import copy
import csv
import random
from typing import Dict, List, Tuple

import torch

from instructions.instructions import KNOWN_TASKS


def sample_batch(data: Dict[str, List[Tuple[List[int], List[int]]]], pad_token_id: int,
                 minibatch: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    tasks = sorted(list(data.keys()))
    if len(tasks) == minibatch:
        tasks_for_batch = copy.copy(tasks)
    elif len(tasks) > minibatch:
        tasks_for_batch = random.sample(population=tasks, k=minibatch)
    else:
        tasks_for_batch = copy.copy(tasks)
        while (minibatch - len(tasks_for_batch)) >= len(tasks):
            tasks_for_batch += tasks
        if (minibatch - len(tasks_for_batch)) > 0:
            tasks_for_batch += random.sample(population=tasks, k=minibatch - len(tasks_for_batch))
    if len(tasks_for_batch) != minibatch:
        err_msg = (f'The minibatch size does not equal to the number of selected tasks. '
                   f'{minibatch} != {len(tasks_for_batch)}')
        raise ValueError(err_msg)
    input_token_ids = []
    input_attention = []
    target_token_ids = []
    target_attention = []
    for cur_task in tasks_for_batch:
        selected_input, selected_target = random.choice(data[cur_task])
        input_token_ids.append(torch.tensor(selected_input, dtype=torch.long))
        input_attention.append(torch.tensor([1 for _ in range(len(selected_input))], dtype=torch.long))
        target_token_ids.append(torch.tensor(selected_target, dtype=torch.long))
        target_attention.append(torch.tensor([1 for _ in range(len(selected_target))], dtype=torch.long))
    input_token_ids_ = torch.nn.utils.rnn.pad_sequence(
        input_token_ids,
        batch_first=True,
        padding_value=pad_token_id
    ).type(torch.LongTensor)
    del input_token_ids
    input_attention_ = torch.nn.utils.rnn.pad_sequence(
        input_attention,
        batch_first=True,
        padding_value=0
    ).type(torch.LongTensor)
    del input_attention
    target_token_ids_ = torch.nn.utils.rnn.pad_sequence(
        target_token_ids,
        batch_first=True,
        padding_value=-100
    ).type(torch.LongTensor)
    del target_token_ids
    target_attention_ = torch.nn.utils.rnn.pad_sequence(
        target_attention,
        batch_first=True,
        padding_value=0
    ).type(torch.LongTensor)
    del target_attention
    return input_token_ids_, input_attention_, target_token_ids_, target_attention_


def load_trainset(fname: str) -> Dict[str, List[Tuple[str, str]]]:
    true_header = ['input', 'target']
    loaded_header = None
    line_idx = 1
    res = dict()
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in data_reader:
            if len(row) > 0:
                err_msg = f'"{fname}": line {line_idx} is wrong!'
                if loaded_header is None:
                    loaded_header = copy.copy(row)
                    if loaded_header != true_header:
                        err_msg += f' The loaded header is impossible! Expected {true_header}, got {loaded_header}.'
                        raise ValueError(err_msg)
                else:
                    if len(row) != len(loaded_header):
                        err_msg += f' The row size is impossible! Expected {len(loaded_header)}, got {len(row)}.'
                        raise ValueError(err_msg)
                    input_text = row[0].strip()
                    target_text = row[1].strip()
                    if not input_text.startswith('<LM>'):
                        err_msg += f' The input text is impossible! It must be started with <LM>. {input_text}'
                        raise ValueError(err_msg)
                    if not target_text.endswith('</s>'):
                        err_msg += f' The target text is impossible! It must be ended with </s>. {target_text}'
                        raise ValueError(err_msg)
                    if len(input_text[4:].strip()) == 0:
                        err_msg += f' The input text is empty! {input_text}'
                        raise ValueError(err_msg)
                    if len(target_text[-4:].strip()) == 0:
                        err_msg += f' The target text is empty! {input_text}'
                        raise ValueError(err_msg)
                    task_id = -1
                    for idx, val in enumerate(KNOWN_TASKS):
                        if input_text[4:].startswith(val[0]):
                            task_id = idx
                            prompt = val[0]
                            context = input_text[(4 + len(val[0])):].strip()
                            while context.startswith('-'):
                                context = context[1:].strip()
                            if len(context) == 0:
                                err_msg += f' The command context is empty! {input_text}'
                                raise ValueError(err_msg)
                            input_text = '<LM>' + prompt.strip() + ' ' + context
                            break
                    if task_id < 0:
                        task_name = 'unknown'
                        instruction = input_text[4:].strip()
                        while instruction.startswith('-'):
                            instruction = instruction[1:].strip()
                        if len(instruction) == 0:
                            err_msg += f' The instruction is empty! {input_text}'
                            raise ValueError(err_msg)
                        input_text = '<LM>' + instruction
                    else:
                        task_name = KNOWN_TASKS[task_id][1]
                    input_text = input_text.replace('\r\n', '\n')
                    target_text = target_text.strip()
                    while target_text.startswith('-'):
                        target_text = target_text[1:].strip()
                    if len(target_text) == 0:
                        err_msg += f' The target is empty! {target_text}'
                        raise ValueError(err_msg)
                    target_text = target_text.replace('\r\n', '\n')
                    if task_name not in res:
                        res[task_name] = []
                    res[task_name].append((input_text, target_text))
            line_idx += 1
    set_of_tasks = set(res.keys())
    if 'detoxification' in set_of_tasks:
        detoxification_prompt = '<LM>Перепиши, пожалуйста, следующий текст так, чтобы он перестал быть токсичным ' \
                                '(неприятным для какой-то группы людей, нарушающим принципы этики).'
        paraphrase_detection_prompt = '<LM>Подскажи, пожалуйста, являются ли парафразами (то есть близкими по ' \
                                      'смыслу) следующие два текста?'
        toxicity_detection_prompt = '<LM>Подскажи, пожалуйста, является ли токсичным (неприятным для какой-то группы ' \
                                    'людей, нарушающим принципы этики) следующий текст?'
        if 'paraphrase_detection' not in set_of_tasks:
            res['paraphrase_detection'] = []
        if 'toxicity_detection' not in set_of_tasks:
            res['toxicity_detection'] = []
        all_indices = set(range(len(res['detoxification'])))
        for idx, (input_text, target_text) in enumerate(res['detoxification']):
            first_text = input_text[len(detoxification_prompt):].strip()
            second_text = target_text[:-len('</s>')].strip()
            res['toxicity_detection'].append((
                toxicity_detection_prompt + ' ' + first_text,
                'Да.</s>'
            ))
            res['toxicity_detection'].append((
                toxicity_detection_prompt + ' ' + second_text,
                'Нет.</s>'
            ))
            res['paraphrase_detection'].append((
                paraphrase_detection_prompt + f' Первый текст: {first_text} Второй текст: {second_text}',
                'Да.</s>'
            ))
            res['paraphrase_detection'].append((
                paraphrase_detection_prompt + f' Первый текст: {second_text} Второй текст: {first_text}',
                'Да.</s>'
            ))
            other_idx = random.choice(list(all_indices - {idx}))
            second_text = res['detoxification'][other_idx][1][:-len('</s>')].strip()
            res['paraphrase_detection'].append((
                paraphrase_detection_prompt + f' Первый текст: {first_text} Второй текст: {second_text}',
                'Нет.</s>'
            ))
            res['paraphrase_detection'].append((
                paraphrase_detection_prompt + f' Первый текст: {second_text} Второй текст: {first_text}',
                'Нет.</s>'
            ))
        del all_indices
    if 'simplification' in set_of_tasks:
        simplification_prompt = '<LM>Упрости, пожалуйста, следующий текст.'
        paraphrase_detection_prompt = '<LM>Подскажи, пожалуйста, являются ли парафразами (то есть близкими по ' \
                                      'смыслу) следующие два текста?'
        if 'paraphrase_detection' not in set_of_tasks:
            res['paraphrase_detection'] = []
        all_indices = set(range(len(res['simplification'])))
        for idx, (input_text, target_text) in enumerate(res['simplification']):
            first_text = input_text[len(simplification_prompt):].strip()
            second_text = target_text[:-len('</s>')].strip()
            res['paraphrase_detection'].append((
                paraphrase_detection_prompt + f' Первый текст: {first_text} Второй текст: {second_text}',
                'Да.</s>'
            ))
            res['paraphrase_detection'].append((
                paraphrase_detection_prompt + f' Первый текст: {second_text} Второй текст: {first_text}',
                'Да.</s>'
            ))
            other_idx = random.choice(list(all_indices - {idx}))
            second_text = res['simplification'][other_idx][1][:-len('</s>')].strip()
            res['paraphrase_detection'].append((
                paraphrase_detection_prompt + f' Первый текст: {first_text} Второй текст: {second_text}',
                'Нет.</s>'
            ))
            res['paraphrase_detection'].append((
                paraphrase_detection_prompt + f' Первый текст: {second_text} Второй текст: {first_text}',
                'Нет.</s>'
            ))
        del all_indices
    list_of_tasks = sorted(list(res.keys()))
    for cur_task in list_of_tasks:
        res[cur_task] = sorted(list(set(res[cur_task])))
    return res
