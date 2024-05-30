from argparse import ArgumentParser
import codecs
import csv
import logging
import os
import random
import sys

import numpy as np
from transformers import T5ForConditionalGeneration, GPT2Tokenizer, GenerationConfig
import torch
from tqdm import tqdm

from inference.inference import generate_answer
from instructions.instructions import KNOWN_TASKS


fredt5_logger = logging.getLogger(__name__)


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)
    device = torch.device('cuda')

    n_processes = max(os.cpu_count(), 1)
    fredt5_logger.info(f'Number of parallel processes is {n_processes}.')

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The input name of FRED-T5.')
    parser.add_argument('-o', '--output', dest='output_fname', type=str, required=True,
                        help='The full path to the output text file with predictions.')
    parser.add_argument('-i', '--input', dest='input_fname', type=str, required=True,
                        help='The full path to the input CSV file with hidden test set.')
    parser.add_argument('--no_lm_tag', dest='no_lm_tag', action='store_true', required=False,
                        help='The <LM> tag is not used.')
    parser.add_argument('--tab', dest='tab', action='store_true', required=False,
                        help='Is tab used as separator instead of comma?')
    parser.add_argument('--dtype', dest='dtype', required=False, default='float32', type=str,
                        choices=['float32', 'float16', 'bfloat16', 'bf16', 'fp16', 'fp32'],
                        help='The PyTorch tensor type for inference.')
    parser.add_argument('--header', dest='input_header', type=str, required=False, default=None,
                        help='The header in the input CSV file.')
    parser.add_argument('-t', '--task', dest='task_type', type=str, choices=['simplification', 'detoxification'],
                        required=False, default='simplification',
                        help='The task type (simplification or detoxification).')
    args = parser.parse_args()

    task_types = [it[1] for it in KNOWN_TASKS]
    task_ID = task_types.index(args.task_type)
    del task_types

    submission_fname: str = os.path.normpath(args.output_fname)
    if not os.path.isfile(submission_fname):
        basedir = os.path.dirname(submission_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                err_msg = f'The directory {basedir} does not exist!'
                fredt5_logger.error(err_msg)
                raise IOError(err_msg)

    input_fname = os.path.normpath(args.input_fname)
    if not os.path.isfile(input_fname):
        err_msg = f'The file {input_fname} does not exist!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)

    fredt5_name = os.path.normpath(args.model_name)
    if not os.path.isdir(fredt5_name):
        err_msg = f'The directory {fredt5_name} does not exist!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)

    if args.dtype in {'float16', 'fp16'}:
        model = T5ForConditionalGeneration.from_pretrained(fredt5_name, torch_dtype=torch.float16).to(device)
    elif args.dtype in {'bfloat16', 'bf16'}:
        model = T5ForConditionalGeneration.from_pretrained(fredt5_name, torch_dtype=torch.bfloat16).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(fredt5_name, torch_dtype=torch.float32).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(fredt5_name)
    generation_config = GenerationConfig.from_pretrained(fredt5_name)
    fredt5_logger.info(f'The instruct model "{os.path.basename(fredt5_name)}" is loaded.')
    fredt5_logger.info(f'{generation_config}'.replace('\n', ' ').replace('\r', ''))

    if args.no_lm_tag:
        question_prefix = ''
    else:
        question_prefix = '<LM>'
    system_prompt = question_prefix + KNOWN_TASKS[task_ID][0] + ' '
    fredt5_logger.info(f'System prompt is: {system_prompt}')
    with codecs.open(input_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        if args.tab:
            csv_reader = csv.reader(fp, delimiter='\t')
        else:
            csv_reader = csv.reader(fp, delimiter=',', quotechar='"')
        if args.input_header is None:
            input_texts = list(map(
                lambda it3: system_prompt + it3[0].strip(),
                filter(
                    lambda it2: len(it2[0].strip()) > 0,
                    filter(lambda it1: len(it1) > 0, csv_reader)
                )
            ))
        else:
            source_rows = list(filter(lambda it: len(it) > 0, csv_reader))
            if args.input_header not in source_rows[0]:
                err_msg = f'The column "{args.input_header}" is not found in the header {source_rows[0]}'
                fredt5_logger.error(err_msg)
                raise ValueError(err_msg)
            col_idx = source_rows[0].index(args.input_header)
            input_texts = list(map(
                lambda it2: system_prompt + it2[col_idx].strip(),
                filter(
                    lambda it1: len(it1[col_idx].strip()) > 0,
                    source_rows[1:]
                )
            ))
    fredt5_logger.info(f'There are {len(input_texts)} input texts. Some of them are:')
    if len(input_texts) > 3:
        selected_texts_to_print = random.sample(population=input_texts, k=3)
    else:
        selected_texts_to_print = input_texts
    for it in selected_texts_to_print:
        fredt5_logger.info(it)
    with codecs.open(submission_fname, mode='w', encoding='utf=8', errors='ignore') as fp:
        for cur_input in tqdm(input_texts):
            answer = generate_answer(
                questions=[cur_input],
                tokenizer=tokenizer,
                model=model,
                config=generation_config
            )[0]
            fp.write(' '.join(answer.strip().split()) + '\n')
    fredt5_logger.info('All texts are simplified.')


if __name__ == '__main__':
    fredt5_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    fredt5_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('fredt5_instruct_evaluation_on_rusimplesenteval.log')
    file_handler.setFormatter(formatter)
    fredt5_logger.addHandler(file_handler)
    main()
