import keras
import tensorflow as tf
import tensorflow_datasets as tfds


def load_lstm_model(path):
    """
    Define a function that loads a model to be explained and returns its instance
    """

    return keras.models.load_model(path)


def preprocess_dataset(dataset, max_sentences_count):
    """
    Counts num of sentences in each instance and filters out those that have less than `max_sentences_count` sentences
    """

    def _proc(x, y):
        return x, len(tf.strings.split(x, sep='.')), y

    dataset = dataset.map(_proc)
    dataset = dataset.filter(lambda x, y, z: y > max_sentences_count)
    dataset = dataset.map(lambda x, y, z: (x, z))

    return dataset


def generate_batches(dataset, num_batches):
    """
    Generator function yielding one batch of dataset at time
    """

    batched = dataset.batch(num_batches)
    for batch in batched.as_numpy_iterator():
        yield batch


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
