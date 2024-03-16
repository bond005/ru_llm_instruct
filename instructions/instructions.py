import copy
import os
import random
from typing import Dict, List, Tuple

from bert_score import bert_cos_score_idf, get_idf_dict
from nltk import wordpunct_tokenize
import numpy as np
from seqeval.metrics import f1_score
from scipy.stats import hmean
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GenerationConfig, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel

from inference.inference import generate_answer, fix_recognition_error
from ner.ner import find_entities_in_text
from utils.utils import levenshtein


KNOWN_TASKS = [
    (
        'Исправь, пожалуйста, ошибки распознавания речи в следующем тексте.',
        'asr_correction'
    ),
    (
        'Выполни саммаризацию и выдели, пожалуйста, основную мысль следующего текста.',
        'summarization'
    ),
    (
        'Разбей, пожалуйста, следующий текст на абзацы.',
        'segmentation'
    ),
    (
        'Упрости, пожалуйста, следующий текст.',
        'simplification'
    ),
    (
        'Найди, пожалуйста, все именованные сущности типа "Организация" в следующем тексте '
        'и выпиши список таких сущностей.',
        'ner_organization'
    ),
    (
        'Найди, пожалуйста, все именованные сущности типа "Человек" в следующем тексте '
        'и выпиши список таких сущностей.',
        'ner_person'
    ),
    (
        'Найди, пожалуйста, все именованные сущности типа "Местоположение" в следующем тексте '
        'и выпиши список таких сущностей.',
        'ner_location'
    )
]


def evaluate_asr_correction(data_for_validation: List[Tuple[str, str]], tokenizer: GPT2Tokenizer,
                            config: GenerationConfig, model: T5ForConditionalGeneration) -> (
        Tuple)[float, List[Tuple[str, str]]]:
    printed_results = []
    n_total_word_dist = 0
    n_total_words = 0
    for input_text, target_text in tqdm(data_for_validation):
        predicted_text = fix_recognition_error(input_text, tokenizer, config, model)
        printed_results.append((predicted_text, target_text))
        target_words = wordpunct_tokenize(target_text)
        cur_dist = levenshtein(wordpunct_tokenize(predicted_text), target_words)
        cur_word_number = len(target_words)
        n_total_word_dist += cur_dist
        n_total_words += cur_word_number
        del target_words
    if len(printed_results) > 5:
        printed_results = random.sample(printed_results, k=5)
    if n_total_words > 0:
        wer = n_total_word_dist / float(n_total_words)
    else:
        wer = 0.0
    return 1.0 - wer, printed_results


def evaluate_segmentation(data_for_validation: List[Tuple[str, str]], tokenizer: GPT2Tokenizer,
                          config: GenerationConfig, model: T5ForConditionalGeneration) -> (
        Tuple)[float, List[Tuple[str, str]]]:
    printed_results = []
    n_total_paragraph_dist = 0
    n_total_paragraphs = 0
    for input_text, target_text in tqdm(data_for_validation):
        predicted_text = generate_answer(input_text, tokenizer, config, model)
        printed_results.append((predicted_text, target_text))
        predicted_paragraphs = list(map(
            lambda it3: ' '.join(list(filter(lambda x: x.isalnum(), wordpunct_tokenize(it3)))).strip(),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), predicted_text.split('\n'))
            )
        ))
        target_paragraphs = list(map(
            lambda it3: ' '.join(list(filter(lambda x: x.isalnum(), wordpunct_tokenize(it3)))).strip(),
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), target_text.split('\n'))
            )
        ))
        cur_dist = levenshtein(predicted_paragraphs, target_paragraphs)
        cur_paragraph_number = len(target_paragraphs)
        n_total_paragraph_dist += cur_dist
        n_total_paragraphs += cur_paragraph_number
    if len(printed_results) > 5:
        printed_results = random.sample(printed_results, k=5)
    if n_total_paragraphs > 0:
        per = n_total_paragraph_dist / float(n_total_paragraphs)
    else:
        per = 0.0
    return 1.0 - per, printed_results


def evaluate_ner(data_for_validation: List[Tuple[str, str]], entity_class: str, tokenizer: GPT2Tokenizer,
                 config: GenerationConfig, model: T5ForConditionalGeneration) -> Tuple[float, List[Tuple[str, str]]]:
    printed_results = []
    prompt_tail = ' и выпиши список таких сущностей.'
    for input_text, target_text in tqdm(data_for_validation):
        found_idx = input_text.find(prompt_tail)
        if found_idx < 0:
            raise ValueError(f'The text "{input_text}" has not a correct prompt!')
        input_text_without_prompt = input_text[(found_idx + len(prompt_tail)):].strip()
        predicted_text = generate_answer(input_text, tokenizer, config, model)
        predicted_text_ = ' '.join(list(filter(lambda x: x.isalnum(), wordpunct_tokenize(predicted_text.lower()))))
        if predicted_text_ == 'в этом тексте нет именованных сущностей такого типа':
            predicted_named_entities = []
        else:
            predicted_named_entities = list(map(
                lambda it3: it3.strip(),
                filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), predicted_text.split('\n'))
                )
            ))
        target_text_ = ' '.join(list(filter(lambda x: x.isalnum(), wordpunct_tokenize(target_text.lower()))))
        if target_text_ == 'в этом тексте нет именованных сущностей такого типа':
            target_named_entities = []
        else:
            target_named_entities = list(map(
                lambda it3: it3.strip(),
                filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), target_text.split('\n'))
                )
            ))
        predicted_named_entities = find_entities_in_text(
            input_text_without_prompt, predicted_named_entities, entity_class
        )
        target_named_entities = find_entities_in_text(
            input_text_without_prompt, target_named_entities, entity_class
        )
        if len(target_named_entities) != len(predicted_named_entities):
            err_msg = (f'The target named entities do not correspond to the text! Text: "{input_text_without_prompt}". '
                       f'Entities: {target_named_entities}')
            raise ValueError(err_msg)
        printed_results.append((predicted_named_entities, target_named_entities))
    y_true = [[x[1] for x in cur[1]] for cur in printed_results]
    y_pred = [[x[1] for x in cur[0]] for cur in printed_results]
    if len(printed_results) > 5:
        printed_results = random.sample(printed_results, k=5)
    f1 = f1_score(y_true, y_pred)
    return f1, printed_results


def evaluate_any_task(data_for_validation: List[Tuple[str, str]], tokenizer: GPT2Tokenizer,
                      config: GenerationConfig, model: T5ForConditionalGeneration,
                      scorer: Tuple[RobertaTokenizer, RobertaModel, int, List[str]]) -> (
        Tuple)[float, List[Tuple[str, str]]]:
    printed_results = []
    candidates = []
    references = []
    texts_for_idf = copy.copy(scorer[3])
    for input_text, target_text in data_for_validation:
        predicted_text = generate_answer(input_text, tokenizer, config, model)
        texts_for_idf.append(predicted_text)
        predicted_tokens = scorer[0].encode(predicted_text, add_special_tokens=True)
        if len(predicted_tokens) > (model.config.max_length - 2):
            predicted_tokens = predicted_tokens[0:(model.config.max_length - 2)]
        target_tokens = scorer[0].encode(target_text, add_special_tokens=True)
        if len(target_tokens) > (model.config.max_length - 2):
            target_tokens = target_tokens[0:(model.config.max_length - 2)]
        candidates.append(scorer[0].decode(predicted_tokens, skip_special_tokens=True))
        references.append(scorer[0].decode(target_tokens, skip_special_tokens=True))
        del predicted_tokens, target_tokens, predicted_text
        printed_results.append((candidates[-1], references[-1]))
    if len(printed_results) > 5:
        printed_results = random.sample(printed_results, k=5)
    idf_dict = get_idf_dict(texts_for_idf, scorer[0], nthreads=max(1, os.cpu_count()))
    del texts_for_idf
    all_preds = bert_cos_score_idf(
        scorer[1],
        references,
        candidates,
        tokenizer,
        idf_dict,
        device=scorer[1].device,
        batch_size=scorer[2],
        all_layers=False
    ).cpu()
    f1_list = all_preds[..., 2].numpy().tolist()
    if len(references) != len(f1_list):
        err_msg = f'The true answers do not correspond to the BERT scores! {len(references)} != {len(f1_list)}.'
        raise ValueError(err_msg)
    f1_mean = float(np.mean(f1_list))
    del idf_dict, f1_list, references, candidates
    return f1_mean, printed_results


def evaluate(data_for_validation: Dict[str, List[Tuple[str, str]]],
             tokenizer: GPT2Tokenizer, config: GenerationConfig, model: T5ForConditionalGeneration,
             intelligent_scorer: Tuple[RobertaTokenizer, RobertaModel, int, List[str]]) -> (
        Tuple)[float, Dict[str, Tuple[float, List[Tuple[str, str]]]]]:
    res = dict()
    scores = []
    for task in data_for_validation:
        if task == 'asr_correction':
            res[task] = evaluate_asr_correction(data_for_validation[task], tokenizer, config, model)
        elif task == 'segmenation':
            res[task] = evaluate_segmentation(data_for_validation[task], tokenizer, config, model)
        elif task.startswith('ner_'):
            entity_class = task[4:].lower()
            res[task] = evaluate_ner(data_for_validation[task], entity_class, tokenizer, config, model)
        else:
            res[task] = evaluate_any_task(data_for_validation[task], tokenizer, config, model, intelligent_scorer)
        scores.append(max(res[task][0], 1e-9))
    mean_score = float(hmean(scores))
    del scores
    return mean_score, res


def load_evaluator(model_path: str, evaluation_batch_size: int,
                   corpus: List[str]) -> Tuple[RobertaTokenizer, RobertaModel, int, List[str]]:
    num_layers = 19
    max_num_layers = 24
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path).cuda()
    model.eval()
    if model.config.num_hidden_layers < max_num_layers:
        err_msg = (f'The RoBERTa model "{os.path.basename(model_path)}" has an inadmissible number of hidden layers. '
                   f'Expected {max_num_layers}, got {model.config.num_hidden_layers}.')
        raise ValueError(err_msg)
    model.encoder.layer = torch.nn.ModuleList(
        [layer for layer in model.encoder.layer[:num_layers]]
    )
    return tokenizer, model, evaluation_batch_size, corpus
