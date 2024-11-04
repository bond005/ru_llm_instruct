from argparse import ArgumentParser
import codecs
import csv
import random
import os
import warnings

from ner.factrueval import load_sample


INSTRUCTIONS_FOR_ORGANIZATIONS = [
    'Найди, пожалуйста, все именованные сущности типа "Организация" в следующем тексте и '
    'выпиши список таких сущностей.',  # 0
    'Прочитай следующий текст и выпиши список всех сущностей класса "Организация".',  # 1
    'Распознай именованные сущности типа "Организация" в тексте и напиши их список '
    '(в каждой строке - отдельная сущность).',  # 2
    'Определи все организации, упоминаемые в тексте, и перечисли их в виде списка.',  # 3
    'Какие организации упоминаются в этом тексте?',  # 4
    'Что за фирмы, учреждения и прочие организации есть в этом тексте? Выпиши их перечень, пожалуйста.',  # 5
    'Найди, пожалуйста, все именованные сущности класса "Организация" в заданном тексте и '
    'напиши перечень найденных сущностей (каждая - с новой строки).',  # 6
]
INSTRUCTIONS_FOR_PERSONS = [
    'Найди, пожалуйста, все именованные сущности типа "Человек" в следующем тексте и '
    'выпиши список таких сущностей.',  # 0
    'Прочитай следующий текст и выпиши список всех сущностей класса "Человек".',  # 1
    'Распознай именованные сущности типа "Человек" в тексте и напиши их список '
    '(в каждой строке - отдельная сущность).',  # 2
    'Определи имена всех людей, упоминаемых в тексте, и перечисли их в виде списка.',  # 3
    'Имена (фамилии и, может быть, отчества) каких людей упоминаются в этом тексте?',  # 4
    'Фамилии и имена (и, возможно, отчества) каких людей обсуждаются в этом тексте? '
    'Выпиши их перечень, пожалуйста.',  # 5
    'Найди, пожалуйста, все именованные сущности класса "Человек" в заданном тексте и '
    'напиши перечень найденных сущностей (каждая - с новой строки).',  # 6
]
INSTRUCTIONS_FOR_LOCATIONS = [
    'Найди, пожалуйста, все именованные сущности типа "Местоположение" в следующем тексте и '
    'выпиши список таких сущностей.',  # 0
    'Прочитай следующий текст и выпиши список всех сущностей класса "Локация".',  # 1
    'Распознай именованные сущности типа "Местоположение" в тексте и напиши их список '
    '(в каждой строке - отдельная сущность).',  # 2
    'Определи все локации, упоминаемые в тексте, и перечисли их в виде списка.',  # 3
    'Какие географические места упоминаются в этом тексте?',  # 4
    'Что за места и локации приведены в этом тексте? Выпиши их перечень, пожалуйста.',  # 5
    'Найди, пожалуйста, все именованные сущности класса "Местоположение" в заданном тексте и '
    'напиши перечень найденных сущностей (каждая - с новой строки).',  # 6
]
NEGATIVE_ANSWER = 'В этом тексте нет именованных сущностей такого типа.'


def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_factrueval_name', type=str, required=True,
                        help='The path to the input FactRuEval-2016.')
    parser.add_argument('-t', '--train', dest='train_fname', type=str, required=True,
                        help='The output CSV file name with instructions for training.')
    parser.add_argument('-e', '--eval', dest='eval_fname', type=str, required=True,
                        help='The output CSV file name with instructions for evaluation.')
    args = parser.parse_args()

    input_dirname = os.path.normpath(args.source_factrueval_name)
    if not os.path.isdir(input_dirname):
        raise ValueError(f'The directory "{input_dirname}" does not exist!')

    output_csv_fname_for_training = os.path.normpath(args.train_fname)
    if not os.path.isfile(output_csv_fname_for_training):
        basedir = os.path.dirname(output_csv_fname_for_training)
        if not os.path.isdir(basedir):
            raise ValueError(f'The directory "{basedir}" does not exist!')

    output_csv_fname_for_validation = os.path.normpath(args.eval_fname)
    if not os.path.isfile(output_csv_fname_for_validation):
        basedir = os.path.dirname(output_csv_fname_for_validation)
        if not os.path.isdir(basedir):
            raise ValueError(f'The directory "{basedir}" does not exist!')

    all_file_items = list(filter(lambda it: it not in {'.', '..'}, os.listdir(input_dirname)))
    if len(all_file_items) == 0:
        raise ValueError(f'The directory "{input_dirname}" is empty!')

    text_files = list(filter(lambda it: it.endswith('.txt'), all_file_items))
    if len(text_files) == 0:
        raise ValueError(f'The directory "{input_dirname}" does not contain any text file!')
    factrueval_samples = [it[:-4] for it in text_files]
    n_training_samples = int(round(0.85 * len(factrueval_samples)))
    if (n_training_samples < 1) or (n_training_samples >= (len(factrueval_samples) - 1)):
        raise ValueError(f'The directory "{input_dirname}" contains a few data!')

    loaded = dict()
    set_of_entities = set()
    for sample in sorted(factrueval_samples):
        if sample in loaded:
            raise ValueError(f'The sample {sample} is duplicated!')
        try:
            loaded[sample] = load_sample(input_dirname, sample)
        except Exception as err:
            warnings.warn(str(err))
        if sample in loaded:
            set_of_entities |= set(loaded[sample][1].keys())
        else:
            factrueval_samples.remove(sample)
    if len(loaded) == 0:
        raise ValueError(f'There are no correct samples in the directory "{input_dirname}".')
    print(f'Named entity classes are: {set_of_entities}')

    random.shuffle(factrueval_samples)
    set_of_entities_for_training = set()
    set_of_entities_for_validation = set()
    for sample in factrueval_samples[0:n_training_samples]:
        set_of_entities_for_training |= set(loaded[sample][1].keys())
    for sample in factrueval_samples[n_training_samples:]:
        set_of_entities_for_validation |= set(loaded[sample][1].keys())
    restarts = 1
    max_restarts = 20
    while (set_of_entities_for_training != set_of_entities) and (set_of_entities_for_validation != set_of_entities):
        random.shuffle(factrueval_samples)
        set_of_entities_for_training = set()
        set_of_entities_for_validation = set()
        for sample in factrueval_samples[0:n_training_samples]:
            set_of_entities_for_training |= set(loaded[sample][1].keys())
        for sample in factrueval_samples[n_training_samples:]:
            set_of_entities_for_validation |= set(loaded[sample][1].keys())
        restarts += 1
        if restarts > max_restarts:
            break
    if restarts > max_restarts:
        raise ValueError(f'The data loaded from "{input_dirname}" cannot be splitted by train and test.')

    samples_for_training = factrueval_samples[0:n_training_samples]
    samples_for_validation = factrueval_samples[n_training_samples:]
    del factrueval_samples
    samples_for_training.sort(key=lambda it: int(it[5:]))
    samples_for_validation.sort(key=lambda it: int(it[5:]))

    with codecs.open(output_csv_fname_for_training, mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['instruction', 'context', 'target'])
        for train_sample in samples_for_training:
            text, entities = loaded[train_sample]
            instruction = random.choice(INSTRUCTIONS_FOR_ORGANIZATIONS)
            if 'organization' in entities:
                if len(entities['organization']) == 0:
                    true_answer = NEGATIVE_ANSWER
                    warnings.warn(f'There are no organizations in the {train_sample}')
                else:
                    true_answer = f'{text[entities["organization"][0][0]:entities["organization"][0][1]]}'
                    for ent_start, ent_end in entities['organization'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
            else:
                true_answer = NEGATIVE_ANSWER
                warnings.warn(f'There are no persons in the {train_sample}')
            if true_answer == NEGATIVE_ANSWER:
                if random.random() > 0.5:
                    data_writer.writerow([instruction, text, true_answer])
            else:
                data_writer.writerow([instruction, text, true_answer])
            del true_answer
            instruction = random.choice(INSTRUCTIONS_FOR_LOCATIONS)
            if 'location' in entities:
                if len(entities['location']) == 0:
                    true_answer = NEGATIVE_ANSWER
                    warnings.warn(f'There are no locations in the {train_sample}')
                else:
                    true_answer = f'{text[entities["location"][0][0]:entities["location"][0][1]]}'
                    for ent_start, ent_end in entities['location'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
            else:
                true_answer = NEGATIVE_ANSWER
                warnings.warn(f'There are no persons in the {train_sample}')
            if true_answer == NEGATIVE_ANSWER:
                if random.random() > 0.5:
                    data_writer.writerow([instruction, text, true_answer])
            else:
                data_writer.writerow([instruction, text, true_answer])
            del true_answer
            instruction = random.choice(INSTRUCTIONS_FOR_PERSONS)
            if 'person' in entities:
                if len(entities['person']) == 0:
                    true_answer = NEGATIVE_ANSWER
                    warnings.warn(f'There are no persons in the {train_sample}')
                else:
                    true_answer = f'{text[entities["person"][0][0]:entities["person"][0][1]]}'
                    for ent_start, ent_end in entities['person'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
            else:
                true_answer = NEGATIVE_ANSWER
                warnings.warn(f'There are no persons in the {train_sample}')
            if true_answer == NEGATIVE_ANSWER:
                if random.random() > 0.5:
                    data_writer.writerow([instruction, text, true_answer])
            else:
                data_writer.writerow([instruction, text, true_answer])
            del text, entities, true_answer

    with codecs.open(output_csv_fname_for_validation, mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['instruction', 'context', 'target'])
        for val_sample in samples_for_validation:
            text, entities = loaded[val_sample]
            instruction = INSTRUCTIONS_FOR_ORGANIZATIONS[0]
            if 'organization' in entities:
                if len(entities['organization']) == 0:
                    true_answer = NEGATIVE_ANSWER
                    warnings.warn(f'There are no organizations in the {val_sample}')
                else:
                    true_answer = f'{text[entities["organization"][0][0]:entities["organization"][0][1]]}'
                    for ent_start, ent_end in entities['organization'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
            else:
                true_answer = NEGATIVE_ANSWER
                warnings.warn(f'There are no persons in the {train_sample}')
            data_writer.writerow([instruction, text, true_answer])
            del true_answer
            instruction = INSTRUCTIONS_FOR_LOCATIONS[0]
            if 'location' in entities:
                if len(entities['location']) == 0:
                    true_answer = NEGATIVE_ANSWER
                    warnings.warn(f'There are no locations in the {val_sample}')
                else:
                    true_answer = f'{text[entities["location"][0][0]:entities["location"][0][1]]}'
                    for ent_start, ent_end in entities['location'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
            else:
                true_answer = NEGATIVE_ANSWER
                warnings.warn(f'There are no persons in the {train_sample}')
            data_writer.writerow([instruction, text, true_answer])
            del true_answer
            instruction = INSTRUCTIONS_FOR_PERSONS[0]
            if 'person' in entities:
                if len(entities['person']) == 0:
                    true_answer = NEGATIVE_ANSWER
                    warnings.warn(f'There are no persons in the {val_sample}')
                else:
                    true_answer = f'{text[entities["person"][0][0]:entities["person"][0][1]]}'
                    for ent_start, ent_end in entities['person'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
            else:
                true_answer = NEGATIVE_ANSWER
                warnings.warn(f'There are no persons in the {train_sample}')
            data_writer.writerow([instruction, text, true_answer])
            del true_answer


if __name__ == '__main__':
    main()
