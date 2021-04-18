import numpy as np

import tme.experiments.experiment_helper as eh
from tme.src import tme

from filelock import FileLock
from io import StringIO
import logging
import time
import ray
import os


@ray.remote
def initialize_task(data, modelpath, fm_it, cnames, workerid):
    """
    Task for each worker with given fm, calculates and logs output
    """

    print(f'Worker id #{workerid} started operating')

    # read model with filelock
    with FileLock(os.path.join(os.path.expanduser(modelpath), "iolock.lock")):
        model = eh.load_lstm_model(os.path.expanduser(modelpath))

    def _predict_proba_fn(_input):
        """
        Function accepting array of instances and returns a probability for each class
        _input - 1d array of instances
        Returns 2d array of [num of instances] x [num of classes] with probabilities
        """
        prediction = model.predict(_input)
        return np.append(prediction, 1 - prediction, axis=1)

    tmexplainer = tme.TextModelsExplainer(classnames=cnames, modelfn=_predict_proba_fn)

    data_x, data_y = data  # unpack data (instances, labels)

    print(f'Worker id #{workerid} processing fm: {fm_it}')

    # create summaries - both
    csummary = tmexplainer.explanation_summaries(data_x, fm=fm_it)
    ssummary = tmexplainer.simple_summaries(data_x)
    print(f'Worker id #{workerid} finished creating summaries for fm: {fm_it}')

    csummary_texts = list(map(lambda x: x[0], csummary))
    ssummary_texts = list(ssummary)

    mcsummaries = model.predict(csummary_texts)
    mssummaries = model.predict(ssummary_texts)
    modelp = model.predict(data_x)

    assert (all([len(modelp) == len(ssummary),
                 len(ssummary) == len(csummary),
                 len(csummary_texts) == len(ssummary),
                 len(mssummaries) == len(ssummary_texts)]))

    # convert true labels
    labels = np.array(list(map(lambda x: [x], data_y)))

    modelp = np.append(modelp, mcsummaries, axis=1)
    modelp = np.append(modelp, mssummaries, axis=1)
    modelp = np.append(modelp, labels, axis=1)

    out = StringIO()
    np.savetxt(
        out,
        modelp,
        fmt='%1.5f',
        header="originalP,customSP,simpleSP,trueClass",
        comments='',
        delimiter=','
    )
    print(f'Worker id #{workerid} finished processing multiplier {fm_it}')

    return out.getvalue()


class FindFactorExperiment:

    def __init__(self, experiment_tag):
        self.experimenttag = experiment_tag
        self.language = "english"
        self.modelpath = "~/school/diploma/src/raw-data/lstm-model-sigmoid"  # path to model on all nodess

        now = time.strftime("%Y-%m-%d_%H:%M")
        self.basepath = f'/home/tomasmizera/school/diploma/src/data/experiments/{experiment_tag}/{now}'
        self.logpath = os.path.join(self.basepath, 'log')
        self.outpath = os.path.join(self.basepath, 'csv')

        # prepare output folder
        os.makedirs(self.basepath, exist_ok=True)
        os.makedirs(self.logpath, exist_ok=True)
        os.makedirs(self.outpath, exist_ok=True)

        # setup logger
        logfilepath = os.path.join(self.logpath, f'{self.experimenttag}-{now}.log')
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(filename=logfilepath)
            ]
        )

        self.log = logging.getLogger()

    def prepare_data(self, batchsize, minnumofsentences):
        data = eh.load_imdb_dataset('train[:100]')
        data = eh.preprocess_dataset(data, minnumofsentences)
        datagen = eh.generate_batches_of_size(data, batchsize)
        datachunk = ''

        try:
            datachunk = next(datagen)
        except StopIteration as e:  # Ran out of data, should not happen
            self._logw(e)
            return 1

        self._logi(f'Data prepared for experiment {self.experimenttag}')
        return datachunk

    def run(self, batch_size=100, to_factor=100, from_sentences_count=10):
        self._logw(f'Starting experiment - {self.experimenttag} ------------------------')

        start = time.time()
        ray.init()

        datachunk = self.prepare_data(batch_size, from_sentences_count)
        if datachunk == 1:
            return 1

        datachunkid = ray.put(datachunk)

        # run workers
        working_ids = []
        fm_mapping = {}  # mapping fm: ray worker id
        for factormultiplier in eh.generate_sequence(to_factor):
            task_id = initialize_task.remote(
                datachunkid,  # id of tuple (list of instances ~ strings, list of labels ~ ints)
                self.modelpath,
                factormultiplier,
                ['Positive', 'Negative'],
                f'work@{factormultiplier}'
            )
            fm_mapping[task_id] = factormultiplier
            working_ids.append(task_id)

        # save done tasks (other workers might still be working)
        while working_ids:
            done_ids, working_ids = ray.wait(working_ids)
            result_id = done_ids[0]

            outfilepath = os.path.join(self.outpath, f'fm-{fm_mapping[result_id]}.csv')

            with open(outfilepath, 'w') as f:
                f.write(ray.get(result_id))
                self._logi(f'Saved data for fm {fm_mapping[result_id]} to {outfilepath}')

        end = time.time()

        self._logi(f'Experiment {self.experimenttag} took {end - start}')
        self._logi(f'Data written to {self.outpath}')
        self._logw(f'Experiment {self.experimenttag} finished -----------------------')

        return 0

    def _logw(self, msg):
        if self.log is not None:
            self.log.warning(msg)

    def _logi(self, msg):
        if self.log is not None:
            self.log.info(msg)

    def _logd(self, msg):
        if self.log is not None:
            self.log.debug(msg)

    def _logc(self, msg):
        if self.log is not None:
            self.log.critical(msg)
