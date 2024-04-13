from argparse import ArgumentParser
import codecs
import csv
import random
import os
import warnings

from ner.factrueval import load_sample


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

    with codecs.open(output_csv_fname_for_training, mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['input', 'target'])
        for train_sample in factrueval_samples[0:n_training_samples]:
            text, entities = loaded[train_sample]
            input_prompt = (f'<LM>Найди, пожалуйста, все именованные сущности типа "Организация" в следующем тексте '
                            f'и выпиши список таких сущностей. {text}')
            if 'organization' in entities:
                if len(entities['organization']) == 0:
                    true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
                else:
                    true_answer = f'{text[entities["organization"][0][0]:entities["organization"][0][1]]}'
                    for ent_start, ent_end in entities['organization'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
                    true_answer += '</s>'
            else:
                true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
            data_writer.writerow([input_prompt, true_answer])
            input_prompt = (f'<LM>Найди, пожалуйста, все именованные сущности типа "Местоположение" в следующем тексте '
                            f'и выпиши список таких сущностей. {text}')
            if 'location' in entities:
                if len(entities['location']) == 0:
                    true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
                else:
                    true_answer = f'{text[entities["location"][0][0]:entities["location"][0][1]]}'
                    for ent_start, ent_end in entities['location'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
                    true_answer += '</s>'
            else:
                true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
            data_writer.writerow([input_prompt, true_answer])
            input_prompt = (f'<LM>Найди, пожалуйста, все именованные сущности типа "Человек" в следующем тексте '
                            f'и выпиши список таких сущностей. {text}')
            if 'person' in entities:
                if len(entities['person']) == 0:
                    true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
                else:
                    true_answer = f'{text[entities["person"][0][0]:entities["person"][0][1]]}'
                    for ent_start, ent_end in entities['person'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
                    true_answer += '</s>'
            else:
                true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
            data_writer.writerow([input_prompt, true_answer])

    with codecs.open(output_csv_fname_for_validation, mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['input', 'target'])
        for val_sample in factrueval_samples[n_training_samples:]:
            text, entities = loaded[val_sample]
            input_prompt = (f'<LM>Найди, пожалуйста, все именованные сущности типа "Организация" в следующем тексте '
                            f'и выпиши список таких сущностей. {text}')
            if 'organization' in entities:
                if len(entities['organization']) == 0:
                    true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
                else:
                    true_answer = f'{text[entities["organization"][0][0]:entities["organization"][0][1]]}'
                    for ent_start, ent_end in entities['organization'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
                    true_answer += '</s>'
            else:
                true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
            data_writer.writerow([input_prompt, true_answer])
            input_prompt = (f'<LM>Найди, пожалуйста, все именованные сущности типа "Местоположение" в следующем тексте '
                            f'и выпиши список таких сущностей. {text}')
            if 'location' in entities:
                if len(entities['location']) == 0:
                    true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
                else:
                    true_answer = f'{text[entities["location"][0][0]:entities["location"][0][1]]}'
                    for ent_start, ent_end in entities['location'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
                    true_answer += '</s>'
            else:
                true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
            data_writer.writerow([input_prompt, true_answer])
            input_prompt = (f'<LM>Найди, пожалуйста, все именованные сущности типа "Человек" в следующем тексте '
                            f'и выпиши список таких сущностей. {text}')
            if 'person' in entities:
                if len(entities['person']) == 0:
                    true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
                else:
                    true_answer = f'{text[entities["person"][0][0]:entities["person"][0][1]]}'
                    for ent_start, ent_end in entities['person'][1:]:
                        true_answer += f'\n{text[ent_start:ent_end]}'
                    true_answer += '</s>'
            else:
                true_answer = 'В этом тексте нет именованных сущностей такого типа.</s>'
            data_writer.writerow([input_prompt, true_answer])


if __name__ == '__main__':
    main()
