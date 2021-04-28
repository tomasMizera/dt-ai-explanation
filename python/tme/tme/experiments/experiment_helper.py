import os
import pickle
import unicodedata
from pathlib import Path

import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from nltk import sent_tokenize
from filelock import FileLock


def load_lstm_model(path):
    """
    Define a function that loads a model to be explained and returns its instance
    """

    return keras.models.load_model(path)


def preprocess_dataset(dataset, max_sentences_count):
    """
    Counts num of sentences in each instance and filters out those that have less than `max_sentences_count` sentences
    """

    @tf.function
    def _proc(x, y):
        return x, len(tf.strings.split(x, sep='.')), y

    dataset = dataset.map(_proc)
    dataset = dataset.filter(lambda x, y, z: y > max_sentences_count)
    dataset = dataset.map(lambda x, y, z: (x, z))

    return dataset


def prepare_dataset_v2(dataset, min_sentences_count, batch_size, expected_size=-1):
    """
    Filters out instances with sentences count lower than min_sentences_count using nltk sent_tokenizer.
    Returns generator
    """

    data = []
    yielded_size = 0

    for text, label in dataset.as_numpy_iterator():
        x = text.decode('utf-8')
        if len(sent_tokenize(x)) < min_sentences_count:
            continue

        data.append((x, label))
        yielded_size += 1

        if expected_size != -1 and yielded_size >= expected_size:
            break  # reached the number of requested instances

        if len(data) >= batch_size:
            yield data
            data.clear()

    if len(data) > 0:
        yield data

    return


def read_precomputed_g(path, batch_size):
    """
    Reads pickles from path and yields them after each one
    """
    dirit = os.scandir(path)
    data = []

    for file in dirit:
        if file.is_file() and file.name.endswith('.pickle'):
            with FileLock(file.path + '.lock'):
                with open(file.path, 'rb') as f:
                    data.extend(pickle.load(f))
            if len(data) >= batch_size:
                yield data
                data.clear()

    if len(data) > 0:
        yield data  # yield the rest that left
    return


def generate_batches(dataset, num_batches):
    """
    Generator function yielding one batch of dataset at time
    """

    batched = dataset.batch(num_batches)
    for batch in batched.as_numpy_iterator():
        yield batch


def load_pickle_object(path):
    """
    Returns loaded pickle object from file, None if loading failed
    Thread safe function
    """

    obj = None
    with FileLock(path + '.lock'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

    return obj


def generate_batches_of_size(dataset, size_of_batch):
    """
    Generator function yielding one batch of fixed size from dataset at time
    """

    seen_instances_count = 0
    while True:
        batch = dataset.skip(seen_instances_count).take(size_of_batch)

        if len(list(batch.as_numpy_iterator())) != size_of_batch:
            raise StopIteration("Ran out of data!")

        # decode strings to UTF-8
        batch = list(map(lambda x: (x[0].decode('UTF-8'), x[1]), batch.as_numpy_iterator()))
        texts = list(map(lambda x: x[0], batch))
        labels = list(map(lambda x: x[1], batch))

        del batch
        seen_instances_count += size_of_batch
        yield texts, labels


def load_imdb_dataset(portion="train+test"):
    """
    Loads TFDS imdb dataset
    """
    return tfds.load(
        name="imdb_reviews",
        split=portion,
        as_supervised=True
    )


def generate_sequence(maxn):
    """
    Generates sequence (list) of slowly growing sequence - each number is incremented by its integer part when devided
    by 10. E.g. 1 -> 1 + ( 1 + 1 / 10 ) = 2 -> ... 10 -> 10 + ( 1 + 10/10 ) = 12 -> ... 70 -> 70 + ( 1 + 70/10 ) = 78
    Sequence ends when new number would be greater than maxn
    """

    seq = []
    i = 0

    while i < maxn:
        seq.append(i)
        i = i + (1 + int(i / 10))

    return seq


# https://www.codegrepper.com/code-examples/python/python+split+dict+into+chunks
def dict_to_chunks(d, outdlen=100):
    from itertools import islice
    it = iter(d)
    for i in range(0, len(d), outdlen):
        yield {k: d[k] for k in islice(it, outdlen)}


def load_files(path_to_files):
    files_it = os.scandir(path_to_files)
    files_contents = []

    for file in files_it:
        if file.is_file() and file.name.endswith('.txt'):
            content = Path(file.path).read_text()
            files_contents.append(content)

    del content
    files_contents = list(map(lambda x: unicodedata.normalize('NFKC', x), files_contents))
    return files_contents
