from argparse import ArgumentParser
import codecs
import copy
import json
import os
import random
from typing import List, Set, Union

from augmentex.char import CharAug
import numpy as np

from instructions.instructions import get_task_type, KNOWN_TASKS, TASK_SYNONYMS


def augment_text(input_text: str, augmentators: List[CharAug], use_lm_tag: bool,
                 existed_texts: Set[str]) -> Union[str, None]:
    selected_augmentator = random.choice(augmentators)
    if get_task_type(input_text, use_lm_tag) == 'asr_correction':
        found_idx = input_text.find('.')
        if found_idx < 0:
            raise RuntimeError(f'The text in incorrect instruction for ASR correction! {input_text}')
        prompt = input_text[0:(found_idx + 1)]
        data = input_text[(found_idx) + 1]
        if (len(prompt) == 0) or (len(data) == 0):
            raise RuntimeError(f'The text in incorrect instruction for ASR correction! {input_text}')
        if use_lm_tag:
            new_input_text = selected_augmentator.augment(prompt[4:])
        else:
            new_input_text = selected_augmentator.augment(prompt)
        new_input_text += data
    else:
        if use_lm_tag:
            new_input_text = selected_augmentator.augment(input_text[4:])
        else:
            new_input_text = selected_augmentator.augment(input_text)
    if new_input_text in existed_texts:
        return None
    if use_lm_tag:
        new_input_text = '<LM>' + new_input_text
    return new_input_text


def main():
    random.seed(42)
    np.random.seed(42)

    char_aug_pc = CharAug(
        unit_prob=0.1,
        min_aug=1,
        max_aug=5,
        mult_num=3,
        lang='rus',
        platform='pc',
        random_seed=42
    )
    char_aug_mobile = CharAug(
        unit_prob=0.1,
        min_aug=1,
        max_aug=5,
        mult_num=3,
        lang='rus',
        platform='mobile',
        random_seed=42
    )

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input name of JSON file with source structured dataset.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of JSON file with structured dataset after filtering.')
    parser.add_argument('--no_lm_tag', dest='no_lm_tag', action='store_true', required=False,
                        help='The <LM> tag is not used.')
    args = parser.parse_args()

    input_dataset_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_dataset_fname):
        err_msg = f'The file {input_dataset_fname} does not exist!'
        raise IOError(err_msg)

    output_dataset_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_dataset_fname):
        base_dir = os.path.dirname(output_dataset_fname)
        if not os.path.isdir(base_dir):
            err_msg = f'The directory {base_dir} does not exist!'
            raise IOError(err_msg)

    with codecs.open(input_dataset_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        err_msg = f'The file "{input_dataset_fname}" contains a wrong data!'
        raise IOError(err_msg)
    if set(data.keys()) != {'text_corpus', 'validation', 'train'}:
        err_msg = f'The file "{input_dataset_fname}" contains a wrong data!'
        raise IOError(err_msg)
    united_text_corpus = data['text_corpus']
    data_for_validation = data['validation']
    data_for_training = data['train']

    tasks_for_training = sorted(list(data_for_training.keys()))
    tasks_for_validation = sorted(list(data_for_validation.keys()))
    print(f'There are {len(tasks_for_validation)} tasks for validation.')
    if set(tasks_for_training) != set(tasks_for_validation):
        err_msg = (f'The training tasks do not correspond to the validation tasks! '
                   f'{tasks_for_training} != {tasks_for_validation}')
        raise ValueError(err_msg)
    for cur_task in tasks_for_training:
        task_samples = copy.copy(data_for_training[cur_task])
        set_of_inputs = set()
        if cur_task == 'unknown':
            for sample in data_for_training[cur_task]:
                if args.no_lm_tag:
                    set_of_inputs.add(sample[0])
                else:
                    set_of_inputs.add(sample[0][4:])
        else:
            for sample in data_for_training[cur_task]:
                task_id = get_task_type(sample[0], use_lm_tag=not args.no_lm_tag)
                if task_id < 0:
                    raise ValueError(f'The task {cur_task}: sample is wrong! {sample}')
                if args.no_lm_tag:
                    lm_tag = ''
                else:
                    lm_tag = '<LM>'
                prompt_text = KNOWN_TASKS[task_id][0]
                instruction_text = sample[0][len(lm_tag + prompt_text):]
                set_of_inputs.add(prompt_text + instruction_text)
                for prompt_synonym in TASK_SYNONYMS[cur_task]['synonyms']:
                    new_input_text = lm_tag + prompt_synonym + instruction_text
                    task_samples.append([new_input_text, sample[1]])
                    set_of_inputs.add(prompt_synonym + instruction_text)
        random.shuffle(task_samples)
        task_samples_ = copy.copy(task_samples)
        for sample in task_samples:
            new_input_text = augment_text(sample[0], [char_aug_pc, char_aug_mobile],
                                          not args.no_lm_tag, set_of_inputs)
            if new_input_text is not None:
                task_samples_.append([
                    new_input_text,
                    sample[1]
                ])
                if args.no_lm_tag:
                    set_of_inputs.add(new_input_text)
                else:
                    set_of_inputs.add(new_input_text[4:])
        del task_samples, data_for_training[cur_task], set_of_inputs
        random.shuffle(task_samples_)
        data_for_training[cur_task] = task_samples_
        del task_samples_

    with codecs.open(output_dataset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(
            fp=fp,
            obj={
                'train': data_for_training,
                'validation': data_for_validation,
                'text_corpus': united_text_corpus
            },
            ensure_ascii=False, indent=4
        )


if __name__ == '__main__':
    main()
