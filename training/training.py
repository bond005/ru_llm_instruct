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
                    input_text = row[0]
                    target_text = row[1]
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
                            break
                    if task_id < 0:
                        task_name = 'unknown'
                    else:
                        task_name = KNOWN_TASKS[task_id][1]
                    if task_name not in res:
                        res[task_name] = []
                    res[task_name].append((input_text, target_text))
    return res
