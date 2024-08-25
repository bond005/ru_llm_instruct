from argparse import ArgumentParser
import codecs
import csv
import random
import os

from tqdm import tqdm

from ner.nerel import load_sample


def main():
    random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_nerel_name', type=str, required=True,
                        help='The path to the input NEREL dataset.')
    parser.add_argument('-d', '--dst', dest='destination_dataset_name', type=str, required=True,
                        help='The path to the resulted HF-formatted instruction dataset.')
    args = parser.parse_args()

    input_dirname = os.path.normpath(args.source_nerel_name)
    if not os.path.isdir(input_dirname):
        raise IOError(f'The directory "{input_dirname}" does not exist!')
    training_dirname = os.path.join(input_dirname, 'train')
    validation_dirname = os.path.join(input_dirname, 'dev')
    test_dirname = os.path.join(input_dirname, 'test')
    if not os.path.isdir(training_dirname):
        raise ValueError(f'The directory "{training_dirname}" does not exist!')
    if not os.path.isdir(validation_dirname):
        raise ValueError(f'The directory "{validation_dirname}" does not exist!')
    if not os.path.isdir(test_dirname):
        raise ValueError(f'The directory "{test_dirname}" does not exist!')

    output_dirname = os.path.normpath(args.destination_dataset_name)
    if not os.path.isdir(output_dirname):
        basedir = os.path.dirname(output_dirname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                raise IOError(f'The directory "{basedir}" does not exist!')
        os.mkdir(output_dirname)
    training_fname = os.path.join(output_dirname, 'train_data.csv')
    validation_fname = os.path.join(output_dirname, 'validation_data.csv')
    test_fname = os.path.join(output_dirname, 'test_data.csv')

    for input_dirname, target_fname in zip([training_dirname, validation_dirname, test_dirname],
                                           [training_fname, validation_fname, test_fname]):
        all_file_items = list(filter(lambda it: it not in {'.', '..'}, os.listdir(input_dirname)))
        if len(all_file_items) == 0:
            raise ValueError(f'The directory "{input_dirname}" is empty!')

        text_files = sorted(list(filter(lambda it: it.endswith('.txt'), all_file_items)))
        if len(text_files) == 0:
            raise ValueError(f'The directory "{input_dirname}" does not contain any text file!')
        data = []
        sources = []
        for it in tqdm(text_files):
            new_samples = load_sample(os.path.join(str(input_dirname), it[:-4]))
            data += new_samples
            sources += [it[:-4] for _ in range(len(new_samples))]
            del new_samples
        with codecs.open(target_fname, mode='w', encoding='utf-8') as fp:
            data_writer = csv.writer(fp, delimiter=',', quotechar='"')
            data_writer.writerow(['input', 'target', 'source'])
            for (inp_txt, ne_labels), cur_source in zip(data, sources):
                data_writer.writerow([inp_txt, '\n'.join([f'{x[0]}\t{x[1]}\t{x[2]}\t{x[3]}' for x in ne_labels]),
                                      cur_source])


if __name__ == '__main__':
    main()
