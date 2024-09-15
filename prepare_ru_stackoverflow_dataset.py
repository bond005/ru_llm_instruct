from argparse import ArgumentParser
import codecs
import csv
import logging
import os
import random
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


RANDOM_SEED: int = 42
ru_stackoverflow_logger = logging.getLogger(__name__)


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input dataset name.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of CSV file with structured and verified RuStackoverflow.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The DeepSeekCoder model name.')
    args = parser.parse_args()

    input_dataset_name = os.path.normpath(args.input_name)
    if not os.path.isdir(input_dataset_name):
        raise IOError(f'The directory {input_dataset_name} does not exist!')

    model_name = os.path.normpath(args.model_name)
    if not os.path.isdir(model_name):
        raise IOError(f'The directory {model_name} does not exist!')

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        basedir = os.path.dirname(output_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                raise IOError(f'The directory "{basedir}" does not exist!')

    dataset = load_dataset(args.input_name, split='train')
    filtered_dataset = dataset.filter(lambda it: it['answer_count'] > 0)

    print(f'There are {len(filtered_dataset)} samples in the source RuStackoverflow dataset.')

    questions_with_answers = []
    questions = [str(it).strip() for it in filtered_dataset['text_markdown']]
    answers = [([str(x1).strip() for x1 in it['text_markdown']], [int(x2) for x2 in it['score']])
               for it in filtered_dataset['answers']]
    del dataset, filtered_dataset

    for question_text, answers_on_question in zip(questions, answers):
        if len(question_text) > 0:
            variants_of_answer = []
            for answer_text, answer_score in zip(answers_on_question[0], answers_on_question[1]):
                if (answer_score > 0) and (len(answer_text) > 0):
                    variants_of_answer.append(
                        {
                            'answer': answer_text,
                            'score': answer_score
                        }
                    )
            if len(variants_of_answer) > 0:
                variants_of_answer.sort(key=lambda it: (-it['score'], -len(it['answer']), it['answer']))
                questions_with_answers.append(
                    {
                        'question': question_text,
                        'answer': variants_of_answer[0]['answer']
                    }
                )
            del variants_of_answer
    print(f'There are {len(questions_with_answers)} samples in the prepared RuStackoverflow dataset.')

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

    prompt = ('You are a programming specialist who also knows Russian. '
              'Please read the question and the answer to this question. '
              'Then think carefully and tell me if this answer is correct, '
              'complete enough and as clear as possible an answer to the question asked? '
              'Just write "yes" or "no", that\'s all.\n\n\n'
              'The question:\n\n\n{question}\n\n\nThe answer to this question:\n\n\n{answer}\n\n\n'
              'Your verdict on the correctness, completeness and clarity of the answer to the question '
              '(only "yes" or "no"): ')
    with codecs.open(output_fname, mode='w', encoding='utf-8', errors='ignore', buffering=0) as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['QUESTION', 'ANSWER', 'VERIFICATION'])
        for sample in tqdm(questions_with_answers):
            messages = [
                {
                    'role': 'user',
                    'content': prompt.format(question=sample['question'], answer=sample['answer'])
                }
            ]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   return_tensors='pt').to(model.device)
            outputs = model.generate(inputs, max_new_tokens=10, do_sample=False, top_k=50, top_p=0.95,
                                     num_return_sequences=1, eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.eos_token_id)
            result = ' '.join(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip().split())
            data_writer.writerow([sample['question'], sample['answer'], result])


if __name__ == '__main__':
    ru_stackoverflow_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    ru_stackoverflow_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('ru_stackoverflow_filter.log')
    file_handler.setFormatter(formatter)
    ru_stackoverflow_logger.addHandler(file_handler)
    main()
