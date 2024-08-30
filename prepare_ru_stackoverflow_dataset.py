from argparse import ArgumentParser
import codecs
import json
import os
import random

from datasets import load_dataset
from sklearn.model_selection import train_test_split


RANDOM_SEED: int = 42


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input dataset name.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of JSON file with structured RuStackoverflow.')
    args = parser.parse_args()

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
                variants_of_answer_without_duplicated = [variants_of_answer[0]]
                for it in variants_of_answer[1:]:
                    if it['score'] != variants_of_answer_without_duplicated[-1]['score']:
                        variants_of_answer_without_duplicated.append(it)
                questions_with_answers.append(
                    {
                        'question': question_text,
                        'answers': variants_of_answer_without_duplicated
                    }
                )
                del variants_of_answer_without_duplicated
            del variants_of_answer
    print(f'There are {len(questions_with_answers)} samples in the prepared RuStackoverflow dataset.')

    questions_with_many_answers = list(filter(lambda it: len(it['answers']) > 1, questions_with_answers))
    print(f'There are {len(questions_with_many_answers)} questions with 2 or more answers.')

    y = [(1 if len(it['answers']) > 1 else 0) for it in questions_with_answers]

    X_train, X_test, y_train, y_test = train_test_split(questions_with_answers, y, test_size=0.1, stratify=y,
                                                        random_state=RANDOM_SEED)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=500, stratify=y_train,
                                                      random_state=RANDOM_SEED)

    result = {
        'train': {
            'question_and_answer': [
                {'question': it['question'], 'answer': it['answers'][0]['answer']}
                for it in X_train
            ],
            'question_and_many_answers': list(filter(lambda it: len(it['answers']) > 1, X_train))
        },
        'validation': {
            'question_and_answer': [
                {'question': it['question'], 'answer': it['answers'][0]['answer']}
                for it in X_val
            ],
            'question_and_many_answers': list(filter(lambda it: len(it['answers']) > 1, X_val))
        },
        'test': {
            'question_and_answer': [
                {'question': it['question'], 'answer': it['answers'][0]['answer']}
                for it in X_test
            ],
            'question_and_many_answers': list(filter(lambda it: len(it['answers']) > 1, X_test))
        }
    }
    with codecs.open(output_fname, mode='w', encoding='utf-8') as fp:
        json.dump(obj=result, fp=fp, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
