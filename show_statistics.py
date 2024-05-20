from argparse import ArgumentParser
import codecs
import json
import os
import random

import numpy as np
from transformers import GPT2Tokenizer


def main():
    random.seed(42)
    np.random.seed(42)

    n_processes = max(os.cpu_count(), 1)
    print(f'Number of parallel processes is {n_processes}.')

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The FRED-T5 model name.')
    parser.add_argument('-d', '--data', dest='data_name', type=str, required=True,
                        help='The JSON file with structured dataset.')
    parser.add_argument('-r', '--result', dest='result_image_name', type=str, required=True,
                        help='The output name of PNG file with sequence length hystogram.')
    parser.add_argument('--split', dest='split', choices=['train', 'validation'], required=True,
                        help='The data split (train or validation).')
    parser.add_argument('--type', dest='type', choices=['input', 'target'], required=True,
                        help='The sample type (input or target).')
    parser.add_argument('--task', dest='task', type=str, required=True, help='The task type.')
    parser.add_argument('--no_lm_tag', dest='no_lm_tag', action='store_true', required=False,
                        help='The <LM> tag is not used.')
    args = parser.parse_args()

    model_path = os.path.normpath(args.model_name)
    if not os.path.isdir(model_path):
        err_msg = f'The directory {model_path} does not exist!'
        raise IOError(err_msg)

    dataset_fname = os.path.normpath(args.data_name)
    if not os.path.isfile(dataset_fname):
        err_msg = f'The file {dataset_fname} does not exist!'
        raise IOError(err_msg)

    output_image_fname = os.path.normpath(args.result_image_name)
    if not os.path.isfile(output_image_fname):
        base_dir = os.path.dirname(output_image_fname)
        if not os.path.isdir(base_dir):
            err_msg = f'The directory {base_dir} does not exist!'
            raise IOError(err_msg)

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    print(f'The pre-trained tokenizer "{os.path.basename(model_path)}" is loaded.')

    with codecs.open(dataset_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        err_msg = f'The file "{dataset_fname}" contains a wrong data!'
        raise IOError(err_msg)
    if set(data.keys()) != {'text_corpus', 'validation', 'train'}:
        err_msg = f'The file "{dataset_fname}" contains a wrong data!'
        raise IOError(err_msg)
    data_for_validation = data['validation']
    data_for_training = data['train']

    n_training_samples = 0
    tasks_for_training = sorted(list(data_for_training.keys()))
    print(f'There are {len(tasks_for_training)} tasks for training.')
    for cur_task in tasks_for_training:
        print(f'There are {len(data_for_training[cur_task])} training samples for task {cur_task}.')
        n_training_samples += len(data_for_training[cur_task])
    print(f'The total number of training samples is {n_training_samples}.')

    tasks_for_validation = sorted(list(data_for_validation.keys()))
    print(f'There are {len(tasks_for_validation)} tasks for validation.')
    if set(tasks_for_training) != set(tasks_for_validation):
        err_msg = (f'The training tasks do not correspond to the validation tasks! '
                   f'{tasks_for_training} != {tasks_for_validation}')
        raise ValueError(err_msg)
    for cur_task in tasks_for_validation:
        print(f'There are {len(data_for_validation[cur_task])} validation samples for task {cur_task}.')

    if args.task not in tasks_for_training:
        raise ValueError(f'The task {args.task} is unknown!')

    texts_for_analysis = []
    if args.split == 'train':
        for sample in data_for_training[args.task]:
            if args.type == 'input':
                if args.no_lm_tag:
                    new_text = sample[0]
                else:
                    new_text = sample[0][4:]
            else:
                new_text = sample[1][:-4]
            texts_for_analysis.append(new_text)
    else:
        for sample in data_for_validation[args.task]:
            if args.type == 'input':
                if args.no_lm_tag:
                    new_text = sample[0]
                else:
                    new_text = sample[0][4:]
            else:
                new_text = sample[1][:-4]
            texts_for_analysis.append(new_text)
    print(f'There are {len(texts_for_analysis)} texts for analysis.')

    pass


if __name__ == '__main__':
    main()
