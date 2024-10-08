from argparse import ArgumentParser
import codecs
import csv
import json
import logging
import os
import random
import sys
from typing import Dict, List, Tuple, Union
from tqdm import tqdm

from gigachat import GigaChat
from nltk import wordpunct_tokenize


gigachat_logger = logging.getLogger(__name__)
CENSORSHIP_RESPONSES = [
    ' не хочу ',
    ' не люблю ',
    ' не знаю ',
    ' не могу ',
    ' не буду ',
    ' не готов ответить ',
    ' не готов отвечать ',
    ' сменим тему ',
    ' сменить тему ',
    ' поговорим о другом ',
    ' поговорим о другой ',
    ' другую тему ',
    ' другая тема ',
    ' другие темы ',
    ' других тем ',
    ' к сожалению ',
    ' к несчастью ',
    ' не указан в ',
    ' менять тему ',
]


def generate_answer_with_gigachat(prompt: str, credentials: str) -> str:
    prepared_prompt = ' '.join(prompt.strip().split())
    if len(prepared_prompt) == 0:
        gigachat_logger.warning('The input prompt is empty!')
        return ''
    prompt_ = ('Ты - полезный ассистент, который хорошо умеет понимать тексты на русском языке и отвечать на вопросы '
               'по этим текстам.\n\n') + prompt
    try:
        with GigaChat(model='GigaChat-Plus', credentials=credentials, scope='GIGACHAT_API_PERS',
                      verify_ssl_certs=False) as giga:
            response = giga.chat(prompt_.strip())
        answer = response.choices[0].message.content.strip()
    except Exception as err:
        answer = ''
        gigachat_logger.warning(str(err))
    return answer


def generate_prompt_for_question_creating(document: str) -> str:
    s = (f'Прочитай, пожалуйста, этот текст и придумай интересный вопрос к этому тексту. '
         f'Вопрос обязательно должен быть таким, чтобы ответ на него требовал знания именно этого текста и '
         f'больше ничего!\n\nТекст: {document}\n\nТвой вопрос: ')
    return s


def generate_prompt_for_answer_creating(document: str, question: str) -> str:
    s = (f'Прочитай, пожалуйста, следующий текст и ответь на вопрос по этому тексту максимально точно и коротко. '
         f'Для ответа на вопрос обязательно используй только знания, полученные при чтении именно этого текста, '
         f'и больше ничего! Если ответа на вопрос нет в тексте, то пиши "Я не знаю."\n\nТекст: {document}\n\n'
         f'Вопрос: {question}\n\nТвой ответ: ')
    return s


def document_to_plain_text(document: Dict[str, Union[str, List[Dict[str, Union[str, List[str]]]]]]) -> str:
    s = ''
    for cur_section in document['document']:
        s +=  '\n\n' + cur_section['section_title'] + '\n\n'
        s += '\n'.join(cur_section['section_body'])
    return s.strip()


def is_censored(text: str) -> bool:
    normalized_text = ' ' + ' '.join(wordpunct_tokenize(text.strip())).replace('ё', 'е') + ' '
    censored = False
    for cur in CENSORSHIP_RESPONSES:
        if normalized_text.find(cur) >= 0:
            censored = True
            break
    return censored


def prepare_positive_and_negative_sample(cur_document: Dict[str, Union[str, List[Dict[str, Union[str, List[str]]]]]],
                                         credentials: str, full_document: bool = True) -> (
        Union)[Tuple[str, str, str], None]:
    selected_section = random.choice(cur_document['document'])
    paragraphs_of_section = ' '.join(selected_section['section_body'])
    question = generate_answer_with_gigachat(
        prompt=generate_prompt_for_question_creating(paragraphs_of_section),
        credentials=credentials
    )
    if len(question.strip()) == 0:
        return None
    if is_censored(question):
        return None
    true_answer = generate_answer_with_gigachat(
        prompt=generate_prompt_for_answer_creating(paragraphs_of_section, question),
        credentials=credentials
    )
    if len(true_answer.strip()) == 0:
        return None
    if is_censored(true_answer):
        return None
    if full_document:
        cur_context = cur_document
    else:
        cur_context = {'document': [selected_section]}
    return question, document_to_plain_text(cur_context), true_answer


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSON with parsed Wiki.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output dataset for RAG.')
    parser.add_argument('-c', '--credentials', dest='credentials', type=str, required=True,
                        help='The credentials for Gigachat API.')
    parser.add_argument('-r', '--random', dest='random_seed', type=int, required=False, default=42,
                        help='The random seed.')
    parser.add_argument('--paragraph', dest='paragraph_only', action='store_true', required=False,
                        help='Paragraph is a unit of context for RAG.')
    args = parser.parse_args()

    random.seed(args.random_seed)

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        err_msg = f'The file "{input_fname}" does not exist!'
        gigachat_logger.error(err_msg)
        raise IOError(err_msg)

    output_dataset_name = os.path.normpath(args.output_name)
    if not os.path.isdir(output_dataset_name):
        basedir = os.path.dirname(output_dataset_name)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                err_msg = f'The directory "{basedir}" does not exist!'
                gigachat_logger.error(err_msg)
                raise IOError(err_msg)
        os.mkdir(output_dataset_name)
    output_trainset_fname = os.path.join(output_dataset_name, 'train_data.csv')
    output_testset_fname = os.path.join(output_dataset_name, 'test_data.csv')
    output_valset_fname = os.path.join(output_dataset_name, 'validation_data.csv')
    if os.path.basename(output_trainset_fname) == os.path.basename(input_fname):
        err_msg = 'The output training file is same as the input file!'
        gigachat_logger.error(err_msg)
        raise ValueError(err_msg)
    if os.path.basename(output_testset_fname) == os.path.basename(input_fname):
        err_msg = 'The output test file is same as the input file!'
        gigachat_logger.error(err_msg)
        raise ValueError(err_msg)

    with codecs.open(input_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    random.shuffle(data)
    gigachat_logger.info(f'{len(data)} samples are loaded.')
    n_test = min(2000, round(0.25 * len(data)))
    n_val = min(1000, round(0.05 * len(data)))
    n_train = len(data) - n_test - n_val
    if (n_train <= 1) or (n_test <= 1) or (n_val <= 1):
        err_msg = f'The dataset "{input_fname}" is too small, and it cannot be split to train and test part.'
        gigachat_logger.error(err_msg)
        raise ValueError(err_msg)

    data_for_training = data[:n_train]
    data_for_validation = data[n_train:(n_train + n_val)]
    data_for_testing = data[(n_train + n_val):]
    del data

    info_msg = (f'There are {len(data_for_training)} samples for training, '
                f'{len(data_for_validation)} samples for validation '
                f'and {len(data_for_testing)} samples for testing.')
    gigachat_logger.info(info_msg)

    if os.path.isfile(output_testset_fname):
        gigachat_logger.warning(f'The "{output_testset_fname}" exists.')
    else:
        all_indices = list(range(len(data_for_testing)))
        counter = 1
        with codecs.open(output_testset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
            data_writer = csv.writer(fp, delimiter=',', quotechar='"')
            data_writer.writerow(['question', 'context', 'response'])
            for idx, val in enumerate(tqdm(data_for_testing)):
                other_idx = random.choice(all_indices)
                while other_idx == idx:
                    other_idx = random.choice(all_indices)
                prepared = prepare_positive_and_negative_sample(
                    cur_document=val,
                    credentials=args.credentials,
                    full_document=not args.paragraph_only
                )
                if prepared is None:
                    warn_msg = f'The document "{val["title"]}" is not processed.'
                    gigachat_logger.warning(warn_msg)
                else:
                    data_writer.writerow([prepared[0], prepared[1], prepared[2]])
                    counter += 1
        gigachat_logger.info(f'{counter} documents from {len(data_for_testing)} are prepared for the test set.')
    del data_for_testing

    if os.path.isfile(output_valset_fname):
        gigachat_logger.warning(f'The "{output_valset_fname}" exists.')
    else:
        all_indices = list(range(len(data_for_validation)))
        counter = 1
        with codecs.open(output_valset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
            data_writer = csv.writer(fp, delimiter=',', quotechar='"')
            data_writer.writerow(['question', 'context', 'response'])
            for idx, val in enumerate(tqdm(data_for_validation)):
                other_idx = random.choice(all_indices)
                while other_idx == idx:
                    other_idx = random.choice(all_indices)
                prepared = prepare_positive_and_negative_sample(
                    cur_document=val,
                    credentials=args.credentials,
                    full_document=not args.paragraph_only
                )
                if prepared is None:
                    warn_msg = f'The document "{val["title"]}" is not processed.'
                    gigachat_logger.warning(warn_msg)
                else:
                    data_writer.writerow([prepared[0], prepared[1], prepared[2]])
                    counter += 1
        gigachat_logger.info(
            f'{counter} documents from {len(data_for_validation)} are prepared for the validation set.')
    del data_for_validation

    if os.path.isfile(output_trainset_fname):
        gigachat_logger.warning(f'The "{output_trainset_fname}" exists.')
    else:
        all_indices = list(range(len(data_for_training)))
        counter = 1
        with codecs.open(output_trainset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
            data_writer = csv.writer(fp, delimiter=',', quotechar='"')
            data_writer.writerow(['question', 'context', 'response'])
            for idx, val in enumerate(tqdm(data_for_training)):
                other_idx = random.choice(all_indices)
                while other_idx == idx:
                    other_idx = random.choice(all_indices)
                prepared = prepare_positive_and_negative_sample(
                    cur_document=val,
                    credentials=args.credentials,
                    full_document=not args.paragraph_only
                )
                if prepared is None:
                    warn_msg = f'The document "{val["title"]}" is not processed.'
                    gigachat_logger.warning(warn_msg)
                else:
                    data_writer.writerow([prepared[0], prepared[1], prepared[2]])
                    counter += 1
        gigachat_logger.info(f'{counter} documents from {len(data_for_training)} are prepared for the training set.')


if __name__ == '__main__':
    gigachat_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    gigachat_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('gigachat_autolabeling.log')
    file_handler.setFormatter(formatter)
    gigachat_logger.addHandler(file_handler)
    main()
