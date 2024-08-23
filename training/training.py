import codecs
import copy
import csv
import json
import logging
import random
from typing import Dict, List, Tuple, Union

import torch

from instructions.instructions import KNOWN_TASKS, get_task_type


training_logger = logging.getLogger(__name__)


def sample_batch(data: Dict[str, List[Tuple[List[int], List[int]]]], pad_token_id: int, minibatch: int,
                 warn: bool = True) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    tasks = sorted(list(data.keys()))
    if 'unknown' in tasks:
        if len(tasks) == 1:
            tasks_for_batch = ['unknown']
        else:
            if minibatch == 1:
                task_weights = [1.0 for _ in range(len(tasks))]
                task_weights[tasks.index('unknown')] = len(tasks) - 1
                weight_sum = sum(task_weights)
                for idx in range(len(tasks)):
                    task_weights[idx] /= weight_sum
                tasks_for_batch = random.choices(population=tasks, k=1, weights=task_weights)
                del task_weights
            else:
                tasks_for_batch = ['unknown' for _ in range(minibatch // 2)]
                tasks.remove('unknown')
                minibatch_ = minibatch - len(tasks_for_batch)
                if len(tasks) == minibatch_:
                    tasks_for_batch_ = copy.copy(tasks)
                elif len(tasks) > minibatch_:
                    tasks_for_batch_ = random.sample(population=tasks, k=minibatch_)
                else:
                    tasks_for_batch_ = copy.copy(tasks)
                    while (minibatch_ - len(tasks_for_batch_)) >= len(tasks):
                        tasks_for_batch_ += tasks
                    if (minibatch_ - len(tasks_for_batch_)) > 0:
                        tasks_for_batch_ += random.sample(population=tasks, k=minibatch_ - len(tasks_for_batch_))
                tasks_for_batch += tasks_for_batch_
                del tasks_for_batch_
    else:
        if warn:
            training_logger.warning(f'The unknown task is not found in the {tasks}.')
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
        training_logger.error(err_msg)
        raise RuntimeError(err_msg)
    random.shuffle(tasks_for_batch)
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


def load_trainset(fname: str, lm_tag: bool = True) -> Dict[str, List[Tuple[str, str]]]:
    true_header = ['input', 'target']
    loaded_header = None
    line_idx = 1
    res = dict()
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, delimiter=',', quotechar='"')
        for row in data_reader:
            if len(row) > 0:
                err_msg = f'"{fname}": line {line_idx} is wrong!'
                line_idx += 1
                if loaded_header is None:
                    loaded_header = copy.copy(row)
                    if loaded_header != true_header:
                        err_msg += f' The loaded header is impossible! Expected {true_header}, got {loaded_header}.'
                        training_logger.error(err_msg)
                        raise IOError(err_msg)
                else:
                    if len(row) != len(loaded_header):
                        err_msg += f' The row size is impossible! Expected {len(loaded_header)}, got {len(row)}.'
                        training_logger.warning(err_msg + f' {row}')
                    else:
                        input_text = row[0].strip()
                        target_text = row[1].strip()
                        if not input_text.startswith('<LM>'):
                            err_msg += f' The input text is impossible! It must be started with <LM>. {input_text}'
                            training_logger.warning(err_msg)
                            continue
                        if not target_text.endswith('</s>'):
                            err_msg += f' The target text is impossible! It must be ended with </s>. {target_text}'
                            training_logger.warning(err_msg)
                            continue
                        if len(input_text[4:].strip()) == 0:
                            err_msg += f' The input text is empty! {input_text}'
                            training_logger.error(err_msg)
                            raise IOError(err_msg)
                        if len(target_text[-4:].strip()) == 0:
                            err_msg += f' The target text is empty! {input_text}'
                            training_logger.error(err_msg)
                            raise IOError(err_msg)
                        task_id = get_task_type(input_text, use_lm_tag=True)
                        if task_id >= 0:
                            task_name = KNOWN_TASKS[task_id][1]
                            prompt = KNOWN_TASKS[task_id][0]
                            context = input_text[(4 + len(prompt)):].strip()
                            while context.startswith('-'):
                                context = context[1:].strip()
                            if len(context) == 0:
                                err_msg += f' The command context is empty! {input_text}'
                                training_logger.error(err_msg)
                                raise IOError(err_msg)
                            input_text = '<LM>' + prompt.strip() + ' ' + context
                        else:
                            task_name = 'unknown'
                            instruction = input_text[4:].strip()
                            while instruction.startswith('-'):
                                instruction = instruction[1:].strip()
                            if len(instruction) == 0:
                                err_msg += f' The instruction is empty! {input_text}'
                                training_logger.error(err_msg)
                                raise IOError(err_msg)
                            input_text = '<LM>' + instruction
                        input_text = input_text.replace('\r\n', '\n')
                        target_text = target_text.strip()
                        while target_text.startswith('-'):
                            target_text = target_text[1:].strip()
                        if len(target_text) == 0:
                            err_msg += f' The target is empty! {target_text}'
                            training_logger.error(err_msg)
                            raise IOError(err_msg)
                        target_text = target_text.replace('\r\n', '\n')
                        if task_name not in res:
                            res[task_name] = []
                        res[task_name].append((input_text, target_text))
            else:
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
        samples_for_task = sorted(list(set(res[cur_task])))
        del res[cur_task]
        if lm_tag:
            res[cur_task] = samples_for_task
        else:
            res[cur_task] = []
            for input_text, target_text in samples_for_task:
                if not input_text.startswith('<LM>'):
                    err_msg = f'The input text is not started from <LM>: {input_text}'
                    training_logger.error(err_msg)
                    raise IOError(err_msg)
                res[cur_task].append((
                    input_text[4:],
                    target_text
                ))
        del samples_for_task
    return res


def create_few_shot_sample(answer: Tuple[str, str], examples: List[Tuple[str, str]],
                           with_system_prompt: bool) -> Tuple[str, str]:
    if len(examples) == 0:
        err_msg = 'The list of examples is empty!'
        training_logger.error(err_msg)
        raise ValueError(err_msg)
    input_question = answer[0]
    true_answer = answer[1]
    use_lm_tag = input_question.startswith('<LM>')
    for example_question, _ in examples:
        use_lm_tag_ = example_question.startswith('<LM>')
        if use_lm_tag:
            if not use_lm_tag_:
                err_msg = ('The input question does not correspond to one of the examples! '
                           'The input question starts from <LM>.')
            else:
                err_msg = ''
        else:
            if use_lm_tag_:
                err_msg = ('The input question does not correspond to one of the examples! '
                           'The input question does not start from <LM>.')
            else:
                err_msg = ''
        if len(err_msg) > 0:
            training_logger.error(err_msg)
            raise ValueError(err_msg)
    task_id = get_task_type(input_question, use_lm_tag=use_lm_tag)
    if task_id < 0:
        err_msg = f'The input question is incorrect! {input_question}'
        training_logger.error(err_msg)
        raise ValueError(err_msg)
    extended_input_question = '<LM>' if use_lm_tag else ''
    if with_system_prompt:
        extended_input_question += KNOWN_TASKS[task_id][0] + ' '
    example_question, example_true_answer = examples[0]
    if use_lm_tag:
        example_question = example_question[(4 + len(KNOWN_TASKS[task_id][0])):].strip()
    else:
        example_question = example_question[len(KNOWN_TASKS[task_id][0]):].strip()
    extended_input_question += 'Вопрос: ' + example_question + '\nОтвет: ' + example_true_answer[:-4]
    for example_question, example_true_answer in examples[1:]:
        if use_lm_tag:
            example_question_ = example_question[(4 + len(KNOWN_TASKS[task_id][0])):].strip()
        else:
            example_question_ = example_question[len(KNOWN_TASKS[task_id][0]):].strip()
        extended_input_question += '\n\nВопрос: ' + example_question_ + '\nОтвет: ' + example_true_answer[:-4]
    extended_input_question += '\n\nВопрос: '
    if use_lm_tag:
        extended_input_question += input_question[(4 + len(KNOWN_TASKS[task_id][0])):].strip()
    else:
        extended_input_question += input_question[len(KNOWN_TASKS[task_id][0]):].strip()
    extended_input_question += '\nОтвет: '
    return extended_input_question, true_answer


def add_few_shot_tasks(few_shot: Dict[str, List[Tuple[str, str]]],
                       questions: Union[Dict[str, List[Tuple[str, str]]], None] = None) -> List[Tuple[str, str]]:
    tasks_for_few_shot_inference = {
        'asr_correction',
        'detoxification',
        'ner_location',
        'ner_person',
        'ner_organization',
        'simplification',
        'toxicity_detection',
        'logical_inference',
        'paraphrase_generation'
    }
    MAX_CHARACTERS_IN_INPUT = 3500
    MIN_SAMPLES_PER_TASK = 6
    set_of_tasks = sorted(list(set(few_shot.keys()) & tasks_for_few_shot_inference))
    if len(set_of_tasks) == 0:
        err_msg = 'There are no tasks for a few-shot inference!'
        training_logger.error(err_msg)
        raise ValueError(err_msg)
    if questions is not None:
        set_of_tasks_ = set(questions.keys()) & tasks_for_few_shot_inference
        if len(set_of_tasks_) == 0:
            err_msg = 'There are no tasks for a few-shot inference!'
            training_logger.error(err_msg)
            raise ValueError(err_msg)
        if not (set_of_tasks_ <= set(set_of_tasks)):
            err_msg = f'The tasks {set_of_tasks_ - set(set_of_tasks)} are excess!'
            training_logger.error(err_msg)
            raise ValueError(err_msg)
        del set_of_tasks
        set_of_tasks = sorted(list(set_of_tasks_))
        del set_of_tasks_
    few_shot_samples = []
    for cur_task in set_of_tasks:
        new_samples = []
        src_short_samples = list(filter(lambda it: len(it[0]) <= MAX_CHARACTERS_IN_INPUT, few_shot[cur_task]))
        if len(src_short_samples) < MIN_SAMPLES_PER_TASK:
            err_msg = f'The task {cur_task} cannot be used for a few-shot inference!'
            training_logger.error(err_msg)
            raise ValueError(err_msg)
        if questions is None:
            for idx, val in enumerate(src_short_samples):
                other_indices = sorted(list(set(range(len(src_short_samples))) - {idx}))
                indices_of_examples = random.sample(
                    population=other_indices,
                    k=random.randint(2, MIN_SAMPLES_PER_TASK - 1)
                )
                if cur_task.startswith('ner_'):
                    samples_for_ner = [src_short_samples[i] for i in indices_of_examples]
                    samples_with_answer = list(filter(
                        lambda it: it[1].lower().find('в этом тексте нет именованных сущностей такого типа') < 0,
                        samples_for_ner
                    ))
                    samples_with_without_answer = list(filter(
                        lambda it: it[1].lower().find('в этом тексте нет именованных сущностей такого типа') >= 0,
                        samples_for_ner
                    ))
                    if len(samples_with_answer) > 0:
                        if len(samples_with_without_answer) > 1:
                            samples_with_without_answer = random.sample(
                                population=samples_with_without_answer,
                                k=1
                            )
                        new_samples.append(create_few_shot_sample(
                            val,
                            samples_with_answer + samples_with_without_answer,
                            False
                        ))
                else:
                    new_samples.append(create_few_shot_sample(
                        val,
                        [src_short_samples[i] for i in indices_of_examples],
                        False
                    ))
        else:
            candidates_to_questions = list(filter(
                lambda it: len(it[0]) <= MAX_CHARACTERS_IN_INPUT,
                questions[cur_task]
            ))
            if len(candidates_to_questions) < MIN_SAMPLES_PER_TASK:
                err_msg = f'The task {cur_task} cannot be used for a few-shot inference!'
                training_logger.error(err_msg)
                raise ValueError(err_msg)
            for val in candidates_to_questions:
                all_indices = list(range(len(src_short_samples)))
                indices_of_examples = random.sample(
                    population=all_indices,
                    k=random.randint(2, MIN_SAMPLES_PER_TASK - 1)
                )
                if cur_task.startswith('ner_'):
                    samples_for_ner = [src_short_samples[i] for i in indices_of_examples]
                    samples_with_answer = list(filter(
                        lambda it: it[1].lower().find('в этом тексте нет именованных сущностей такого типа') < 0,
                        samples_for_ner
                    ))
                    samples_with_without_answer = list(filter(
                        lambda it: it[1].lower().find('в этом тексте нет именованных сущностей такого типа') >= 0,
                        samples_for_ner
                    ))
                    if len(samples_with_answer) > 0:
                        if len(samples_with_without_answer) > 1:
                            samples_with_without_answer = random.sample(
                                population=samples_with_without_answer,
                                k=1
                            )
                        new_samples.append(create_few_shot_sample(
                            val,
                            samples_with_answer + samples_with_without_answer,
                            False
                        ))
                else:
                    new_samples.append(create_few_shot_sample(
                        val,
                        [src_short_samples[i] for i in indices_of_examples],
                        False
                    ))
        if len(new_samples) > 0:
            info_msg = f'Task {cur_task}: {len(new_samples)} few-shot samples are added.'
            training_logger.info(info_msg)
            if len(new_samples) > 2:
                selected_samples_for_print = random.sample(population=new_samples, k=2)
            else:
                selected_samples_for_print = new_samples
            for cur in selected_samples_for_print:
                training_logger.info(json.dumps(obj=cur, ensure_ascii=False))
            few_shot_samples += new_samples
        del new_samples, src_short_samples
    return few_shot_samples
