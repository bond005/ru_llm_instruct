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
    task_types = [it[1] for it in KNOWN_TASKS]
    task_ID = task_types.index('simplification')
    del task_types

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
    parser.add_argument('--dtype', dest='dtype', required=False, default='float32', type=str,
                        choices=['float32', 'float16', 'bfloat16', 'bf16', 'fp16', 'fp32'],
                        help='The PyTorch tensor type for inference.')
    args = parser.parse_args()

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
        csv_reader = csv.reader(fp, delimiter=',', quotechar='"')
        input_texts = list(map(
            lambda it3: system_prompt + it3[0].strip(),
            filter(
                lambda it2: len(it2[0].strip()) > 0,
                filter(lambda it1: len(it1) > 0, csv_reader)
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
