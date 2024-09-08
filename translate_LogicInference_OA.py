from argparse import ArgumentParser
import codecs
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

logic_inference_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        logic_inference_logger.error(err_msg)
        raise ValueError(err_msg)
    device = torch.device('cuda')
    torch.cuda.manual_seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input name of LogicInference_OA dataset in English.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of LogicInference_OA dataset in Russian.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The path to NLLB model.')
    args = parser.parse_args()

    source_dataset_path = os.path.normpath(args.input_name)
    if not os.path.isdir(source_dataset_path):
        err_msg = f'The directory "{source_dataset_path}" does not exist!'
        logic_inference_logger.error(err_msg)
        raise IOError(err_msg)
    source_dataset_fname = os.path.join(source_dataset_path, 'train.csv')
    if not os.path.isfile(source_dataset_fname):
        err_msg = f'The file "{source_dataset_fname}" does not exist!'
        logic_inference_logger.error(err_msg)
        raise IOError(err_msg)

    destination_dataset_path = os.path.normpath(args.output_name)
    if not os.path.isdir(destination_dataset_path):
        err_msg = f'The directory "{destination_dataset_path}" does not exist!'
        logic_inference_logger.error(err_msg)
        raise IOError(err_msg)
    if os.path.basename(source_dataset_path) == os.path.basename(destination_dataset_path):
        err_msg = f'The destination dataset path is equal to the source one!'
        logic_inference_logger.error(err_msg)
        raise IOError(err_msg)

    destination_dataset_fname = os.path.join(destination_dataset_path, 'train.csv')

    model_name = os.path.normpath(args.model_name)
    if not os.path.isdir(model_name):
        err_msg = f'The directory "{model_name}" does not exist!'
        logic_inference_logger.error(err_msg)
        raise IOError(err_msg)

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang='eng_Latn')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    config = GenerationConfig.from_pretrained(model_name)
    max_en_text_length = round(config.max_length * 0.7)
    if config.num_beams is None:
        config.num_beams = 3
    elif config.num_beams < 3:
        config.num_beams = 3

    with codecs.open(source_dataset_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_samples_with_header = list(filter(
            lambda it3: (len(it3[0]) > 0) and (len(it3[1]) > 0) and
                        (len(tokenizer.tokenize(it3[0])) < max_en_text_length) and
                        (len(tokenizer.tokenize(it3[1])) < max_en_text_length),
            map(
                lambda it2: (it2[0].strip(), it2[1].strip(), it2[2].strip()),
                filter(lambda it1: len(it1) == 3, data_reader)
            )
        ))
    true_header = ('INSTRUCTION', 'RESPONSE', 'SOURCE')
    if data_samples_with_header[0] != ('INSTRUCTION', 'RESPONSE', 'SOURCE'):
        err_msg = (f'The file "{source_dataset_fname}" contains a wrong header! '
                   f'Expected {true_header}, got {data_samples_with_header[0]}.')
        logic_inference_logger.error(err_msg)
        raise IOError(err_msg)
    logic_inference_logger.info(f'There are {len(data_samples_with_header) - 1} samples.')

    counter = 1
    with codecs.open(destination_dataset_fname, mode='w', encoding='utf-8', buffering=0) as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(list(true_header) + ['TRANSLATION_SCORE'])
        for instruction_en, response_en, source in data_samples_with_header[1:]:
            inputs = tokenizer(instruction_en, return_tensors='pt').to(device)
            generated = model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id['rus_Cyrl'],
                generation_config=config, return_dict_in_generate=True, output_scores=True
            )
            instruction_ru = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0]
            instruction_score = float(generated.sequences_scores.cpu().numpy()[0])
            del inputs, generated
            inputs = tokenizer(response_en, return_tensors='pt').to(device)
            generated = model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id['rus_Cyrl'],
                generation_config=config, return_dict_in_generate=True, output_scores=True
            )
            response_ru = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0]
            response_score = float(generated.sequences_scores.cpu().numpy()[0])
            del inputs, generated
            united_score = min(response_score, instruction_score)
            data_writer.writerow([instruction_ru, response_ru, source, str(round(united_score, 6))])
            if counter % 500 == 0:
                info_msg = f'{counter} samples from {len(data_samples_with_header) - 1} are translated.'
                logic_inference_logger.info(info_msg)
            counter += 1
    if counter != len(data_samples_with_header):
        info_msg = '{counter - 1} samples from {len(data_samples_with_header) - 1} are translated.'
        logic_inference_logger.info(info_msg)


if __name__ == '__main__':
    logic_inference_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logic_inference_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('logic_inference_translation.log')
    file_handler.setFormatter(formatter)
    logic_inference_logger.addHandler(file_handler)
    main()
