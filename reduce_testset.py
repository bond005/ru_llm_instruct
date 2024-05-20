from argparse import ArgumentParser
import codecs
import json
import math
import os
import random

import numpy as np


def main():
    random.seed(42)
    np.random.seed(42)

    n_processes = max(os.cpu_count(), 1)
    print(f'Number of parallel processes is {n_processes}.')

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input name of JSON file with source structured dataset.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of JSON file with structured dataset after filtering.')
    parser.add_argument('--testsize', dest='testsize', type=int, required=True,
                        help='The maximal number of validation samples per validated task.')
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

    if args.testsize < 6:
        err_msg = (f'The maximal number of validation samples per validated task is too small! '
                   f'Expected 6 or greater, got {args.testsize}.')
        raise ValueError(err_msg)

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
    for cur_task in tasks_for_validation:
        if cur_task.startswith('ner_'):
            samples_with_ne = list(filter(
                lambda it: it[1].find('нет именованных сущностей такого типа') < 0,
                data_for_validation[cur_task]
            ))
            samples_without_ne = list(filter(
                lambda it: it[1].find('нет именованных сущностей такого типа') >= 0,
                data_for_validation[cur_task]
            ))
            if (len(samples_with_ne) + len(samples_without_ne)) != len(data_for_validation[cur_task]):
                err_msg = f'The data for task {cur_task} is incorrect!'
                raise ValueError(err_msg)
            if len(samples_with_ne) == 0:
                err_msg = f'The data for task {cur_task} is incorrect!'
                raise ValueError(err_msg)
            if len(samples_without_ne) == 0:
                err_msg = f'The data for task {cur_task} is incorrect!'
                raise ValueError(err_msg)
            if len(data_for_validation[cur_task]) > args.testsize:
                if len(samples_with_ne) > math.ceil(args.testsize / 2):
                    samples_for_ner = random.sample(samples_with_ne, k=math.ceil(args.testsize / 2))
                else:
                    samples_for_ner = samples_with_ne
                if len(samples_without_ne) > (args.testsize - len(samples_for_ner)):
                    samples_for_ner += random.sample(samples_without_ne, k=args.testsize - len(samples_for_ner))
                else:
                    samples_for_ner += samples_without_ne
                del data_for_validation[cur_task]
                data_for_validation[cur_task] = samples_for_ner
                del samples_for_ner
            del samples_with_ne, samples_without_ne
        else:
            if len(data_for_validation[cur_task]) > args.testsize:
                data_for_validation[cur_task] = random.sample(data_for_validation[cur_task], k=args.testsize)

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
