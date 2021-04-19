from lime import lime_text
from tme.experiments import experiment_helper as eh
import time
import numpy as np
import ray


@ray.remote
def do_iteration(uuid, testtype, data):
    start = time.time()
    print(f"Iteration@{uuid} started")

    model = eh.load_lstm_model("/home/tomasmizera/school/diploma/src/raw-data/lstm-model-sigmoid")
    print(f"Iteration@{uuid} t1 {time.time() - start}")

    def _predict_proba_fn(_input):
        """
        Function accepting array of instances and returns a probability for each class
        _input - 1d array of instances
        Returns 2d array of [num of instances] x [num of classes] with probabilities
        """
        strt = time.time()
        prediction = model.predict(_input)
        outarr = np.append(prediction, 1 - prediction, axis=1)

        return outarr

    explainer = lime_text.LimeTextExplainer(class_names=['Positive', 'Negative'])
    print(f"Iteration@{uuid} t2 {time.time() - start}")

    compstart = time.time()

    maxn = len(data)

    if testtype == 'A':

        for i in range(maxn):
            explainer.explain_instance(data[i], _predict_proba_fn, num_features=100)

    elif testtype == 'B':

        for i in range(maxn):
            explainer = lime_text.LimeTextExplainer(class_names=['Positive', 'Negative'])
            explainer.explain_instance(data[i], _predict_proba_fn, num_features=100)

    else:
        raise TypeError ("No such test type")

    compend = time.time() - compstart
    print(f'Iteration@{uuid} computation took {compend} secs, per iteration approx {compend/maxn}')


def run():
    data = eh.load_imdb_dataset('train+test')
    data = eh.preprocess_dataset(data, 40)
    datagen = eh.generate_batches_of_size(data, 20)
    data_x, data_y = next(datagen)

    ray.init(address='auto', _redis_password='5241590000000000')
    aid = do_iteration.remote(0, 'A', data_x)
    bid = do_iteration.remote(1, 'B', data_x)
    ray.get([aid, bid])

