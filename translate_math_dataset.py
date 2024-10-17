from argparse import ArgumentParser
import codecs
import copy
import csv
import logging
import os
import random
import re
import sys
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizer, GenerationConfig


math_instruct_logger = logging.getLogger(__name__)
RANDOM_SEED: int = 42


def levenshtein(seq1: List[str], seq2: List[str]) -> float:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    int(matrix[x - 1, y]) + 1,
                    int(matrix[x - 1, y - 1]),
                    int(matrix[x, y - 1]) + 1
                )
            else:
                matrix[x, y] = min(
                    int(matrix[x - 1, y]) + 1,
                    int(matrix[x - 1, y - 1]) + 1,
                    int(matrix[x, y - 1]) + 1
                )
    return float(matrix[size_x - 1, size_y - 1])


def calculate_lexical_similarity(ref: str, cand: str) -> float:
    return max(0.0, 1.0 - levenshtein(list(ref), list(cand)) / len(ref))


def prepare_numbers(source_text: str) -> str:
    re_for_russian_number = re.compile(r'[\s^]+\d+,\d+[\s$]+')
    new_text = copy.copy(source_text)
    search_res = re_for_russian_number.search(new_text)
    while search_res is not None:
        if (search_res.start() < 0) or (search_res.end() <= search_res.start()):
            break
        bounds = (
            search_res.start(),
            search_res.end()
        )
        new_text = (new_text[:bounds[0]] + new_text[bounds[0]:bounds[1]].replace(',', '.') +
                    new_text[bounds[1]:])
        search_res = re_for_russian_number.search(new_text)
    return new_text


def select_variants(s: str) -> List[str]:
    re_for_variant = re.compile(r'\(\w\)')
    variants = []
    start_pos = 0
    search_res = re_for_variant.search(s)
    while search_res is not None:
        if (search_res.start() < 0) or (search_res.end() <= search_res.start()):
            break
        bounds = (
            search_res.start() + start_pos,
            search_res.end() + start_pos
        )
        variants.append(s[(bounds[0] + 1):(bounds[1] - 1)])
        start_pos = bounds[1]
        search_res = re_for_variant.search(s[start_pos:])
    return variants


def translate_line(english_text: str, nmt_tokenizer: NllbTokenizer, nmt_model: M2M100ForConditionalGeneration,
                   nmt_generation: GenerationConfig) -> Tuple[str, float]:
    inputs = nmt_tokenizer(english_text, return_tensors='pt').to(nmt_model.device)
    generated = nmt_model.generate(
        **inputs, forced_bos_token_id=nmt_tokenizer.lang_code_to_id['rus_Cyrl'],
        generation_config=nmt_generation, return_dict_in_generate=True, output_scores=True
    )
    russian_text = nmt_tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0]
    translation_score = float(generated.sequences_scores.cpu().numpy()[0])
    del inputs, generated
    return russian_text, translation_score


def translate_text(english_text: str, nmt_tokenizer: NllbTokenizer, nmt_model: M2M100ForConditionalGeneration,
                   nmt_generation: GenerationConfig, is_python: bool,
                   embedder: SentenceTransformer) -> Tuple[str, float, float, float]:
    lines_of_english_text = english_text.replace('\r', '').split('\n')
    if len(list(filter(lambda x: len(x.strip()) > 0, lines_of_english_text))) > 1:
        lines_of_russian_text = []
        scores = []
        for cur_line in lines_of_english_text:
            prepared_line = cur_line.strip()
            start_pos = cur_line.find(prepared_line)
            spaces_before = ''.join([' ' for _ in range(start_pos)])
            spaces_after = ''.join([' ' for _ in range(len(cur_line) - (len(prepared_line) + start_pos))])
            if prepared_line.startswith('#') or (not is_python):
                if prepared_line.startswith('#'):
                    prepared_line_ = prepared_line[1:].strip()
                    start_pos = prepared_line.find(prepared_line_)
                    spaces_before_ = ''.join([' ' for _ in range(start_pos)])
                    spaces_after_ = ''.join([' ' for _ in range(len(cur_line) - (len(prepared_line_) + start_pos))])
                    translated, score = translate_line(prepared_line[1:], nmt_tokenizer, nmt_model, nmt_generation)
                    new_line = spaces_before
                    new_line += '#' + spaces_before_ + translated + spaces_after_
                    new_line += spaces_after
                else:
                    translated, score = translate_line(prepared_line, nmt_tokenizer, nmt_model, nmt_generation)
                    new_line = spaces_before
                    new_line += translated
                    new_line += spaces_after
                lines_of_russian_text.append(new_line)
                scores.append(score)
            else:
                lines_of_russian_text.append(cur_line)
                scores.append(0.0)
        translation_score = min(scores)
        del scores
        russian_text = prepare_numbers('\n'.join(lines_of_russian_text))
        russian_text_v2, other_score = translate_line(english_text, nmt_tokenizer, nmt_model, nmt_generation)
        translation_score = min(translation_score, other_score)
        embeddings = embedder.encode([russian_text, russian_text_v2], normalize_embeddings=True)
        semantic_homogeneous = 0.0
        for idx in range(embeddings.shape[1]):
            semantic_homogeneous += embeddings[0, idx] * embeddings[1, idx]
        semantic_homogeneous += 1.0
        semantic_homogeneous /= 2.0
        lexical_homogeneous = calculate_lexical_similarity(russian_text, russian_text_v2)
        del lines_of_russian_text
        is_answer_choice = (' '.join(english_text.split()).strip().lower().find('answer choice') >= 0)
        if is_answer_choice:
            english_variants = select_variants(english_text)
            russian_variants = select_variants(russian_text)
            if english_variants != russian_variants:
                semantic_homogeneous = 0.0
                lexical_homogeneous = 0.0
    else:
        russian_text_v2, translation_score = translate_line(english_text, nmt_tokenizer, nmt_model, nmt_generation)
        russian_text = prepare_numbers(russian_text_v2)
        semantic_homogeneous = 1.0
        lexical_homogeneous = calculate_lexical_similarity(russian_text, russian_text_v2)

    return russian_text, translation_score, semantic_homogeneous, lexical_homogeneous


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        math_instruct_logger.error(err_msg)
        raise ValueError(err_msg)
    device = torch.device('cuda')
    torch.cuda.manual_seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input name of LogicInference_OA dataset in English.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of LogicInference_OA dataset in Russian.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=False,
                        default='facebook/nllb-200-3.3B', help='The path to the NLLB model.')
    parser.add_argument('-e', '--embedder', dest='embedder_name', type=str, required=False,
                        default='intfloat/multilingual-e5-large', help='The path to the embedder.')
    args = parser.parse_args()

    source_dataset_path = os.path.normpath(args.input_name)
    if not os.path.isdir(source_dataset_path):
        err_msg = f'The directory "{source_dataset_path}" does not exist!'
        math_instruct_logger.error(err_msg)
        raise IOError(err_msg)
    source_dataset_fname = os.path.join(source_dataset_path, 'train.csv')
    if not os.path.isfile(source_dataset_fname):
        err_msg = f'The file "{source_dataset_fname}" does not exist!'
        math_instruct_logger.error(err_msg)
        raise IOError(err_msg)

    destination_dataset_path = os.path.normpath(args.output_name)
    if not os.path.isdir(destination_dataset_path):
        err_msg = f'The directory "{destination_dataset_path}" does not exist!'
        math_instruct_logger.error(err_msg)
        raise IOError(err_msg)
    if os.path.basename(source_dataset_path) == os.path.basename(destination_dataset_path):
        err_msg = f'The destination dataset path is equal to the source one!'
        math_instruct_logger.error(err_msg)
        raise IOError(err_msg)

    destination_dataset_fname = os.path.join(destination_dataset_path, 'train.csv')

    math_instruct_logger.info(f'Neural machine translator name is {args.model_name}')
    try:
        tokenizer = NllbTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            src_lang='eng_Latn', tgt_lang='rus_Cyrl'
        )
        model_name = args.model_name
    except:
        model_name = os.path.normpath(args.model_name)
        try:
            tokenizer = NllbTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name,
                src_lang='eng_Latn', tgt_lang='rus_Cyrl'
            )
        except Exception as err:
            math_instruct_logger.error(str(err))
            raise
    try:
        model = M2M100ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2'
        ).to(device)
    except Exception as err:
        math_instruct_logger.warning(str(err))
        try:
            model = M2M100ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_name,
                torch_dtype=torch.bfloat16
            ).to(device)
        except Exception as err:
            math_instruct_logger.error(str(err))
            raise
    model.eval()
    config = GenerationConfig.from_pretrained(model_name)
    max_en_text_length = round(config.max_length * 0.7)
    if config.num_beams is None:
        config.num_beams = 3
    elif config.num_beams < 3:
        config.num_beams = 3

    try:
        embedder = SentenceTransformer(args.embedder_name)
    except Exception as err:
        math_instruct_logger.error(str(err))
        raise

    with codecs.open(source_dataset_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_samples_with_header = list(filter(
            lambda it3: (len(it3[0].strip()) > 0) and (len(it3[1].strip()) > 0) and
                        (len(tokenizer.tokenize(it3[0])) < max_en_text_length) and
                        (len(tokenizer.tokenize(it3[1])) < max_en_text_length),
            map(lambda it2: (it2[0], it2[1], it2[2]), filter(lambda it1: len(it1) == 3, data_reader))
        ))
    true_header = ('INSTRUCTION', 'RESPONSE', 'SOURCE')
    if data_samples_with_header[0] != ('INSTRUCTION', 'RESPONSE', 'SOURCE'):
        err_msg = (f'The file "{source_dataset_fname}" contains a wrong header! '
                   f'Expected {true_header}, got {data_samples_with_header[0]}.')
        math_instruct_logger.error(err_msg)
        raise IOError(err_msg)
    math_instruct_logger.info(f'There are {len(data_samples_with_header) - 1} samples.')

    counter = 1
    with codecs.open(destination_dataset_fname, mode='w', encoding='utf-8', buffering=0) as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['INSTRUCTION_EN', 'RESPONSE_EN', 'INSTRUCTION_RU', 'RESPONSE_RU', 'TRANSLATION_SCORE',
                              'SEMANTIC_HOMOGENEITY', 'LEXICAL_HOMOGENEITY'])
        for instruction_en, response_en, source in data_samples_with_header[1:]:
            is_python = ((' '.join(instruction_en.split()).strip().find(' Python') >= 0) or
                         (' '.join(instruction_en.split()).strip().find(' program ') >= 0))
            instruction_ru, instruction_score, sem_sim_1, lex_sim_1 = translate_text(
                instruction_en,
                tokenizer, model, config,
                False,
                embedder
            )
            response_ru, response_score, sem_sim_2, lex_sim_2 = translate_text(
                response_en,
                tokenizer, model, config,
                is_python,
                embedder
            )
            united_score = min(response_score, instruction_score)
            semantic_similarity = min(sem_sim_1, sem_sim_2)
            lexical_similarity = min(lex_sim_1, lex_sim_2)
            data_writer.writerow([instruction_en, response_en, instruction_ru, response_ru, str(round(united_score, 6)),
                                  str(round(semantic_similarity, 6)), str(round(lexical_similarity, 6))])
            if counter % 500 == 0:
                info_msg = f'{counter} samples from {len(data_samples_with_header) - 1} are translated.'
                math_instruct_logger.info(info_msg)
            counter += 1
    if counter != len(data_samples_with_header):
        info_msg = '{counter - 1} samples from {len(data_samples_with_header) - 1} are translated.'
        math_instruct_logger.info(info_msg)


if __name__ == '__main__':
    math_instruct_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    math_instruct_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('math_instruct_translation.log')
    file_handler.setFormatter(formatter)
    math_instruct_logger.addHandler(file_handler)
    main()
