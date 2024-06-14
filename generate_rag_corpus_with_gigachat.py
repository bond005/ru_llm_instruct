from argparse import ArgumentParser
import logging
import os
import sys

from gigachat import GigaChat


gigachat_logger = logging.getLogger(__name__)


def generate_answer_with_gigachat(prompt: str, credentials: str) -> str:
    prepared_prompt = ' '.join(prompt.strip().split())
    if len(prepared_prompt) == 0:
        gigachat_logger.warning('The input prompt is empty!')
        return ''
    with GigaChat(credentials=credentials, scope='GIGACHAT_API_PERS', verify_ssl_certs=False) as giga:
        response = giga.chat(prompt.strip())
    return response.choices[0].message.content.strip()


def generate_prompt_for_question_creating(document: str) -> str:
    s = (f'Прочитай, пожалуйста, этот текст и придумай интересный вопрос к этому тексту. '
         f'Вопрос обязательно должен быть таким, чтобы ответ на него требовал знания именно этого текста и '
         f'больше ничего!\n\nТекст: {document}\n\nТвой вопрос: ')
    return s


def generate_prompt_for_answer_creating(document: str, question: str) -> str:
    s = (f'Прочитай, пожалуйста, следующий текст и ответь на вопрос по этому тексту максимально точно и коротко. '
         f'Для ответа на вопрос обязательно используй именно текст и больше ничего!\n\n'
         f'Текст: {document}\n\nВопрос: {question}\n\nТвой ответ: ')
    return s


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSON with parsed Wiki.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output dataset for RAG.')
    parser.add_argument('-c', '--credentials', dest='credentials', type=str, required=True,
                        help='The credentials for Gigachat API.')
    args = parser.parse_args()

    pass


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
