import sys

import numpy as np

import tme.experiments.experiment_helper as eh
from tme.src import tme

from filelock import FileLock
from lime import lime_text
from io import StringIO
import logging
import pickle
import time
import ray
import os


@ray.remote
def initialize_task(data, factor_multiplier, modelpath, workerid):
    """
    Task for each worker with given fm, calculates and logs output
    """

    print(f'Worker id #{workerid} started operating')

    tmexplainer = tme.TextModelsExplainer(uuid=workerid)

    print(f'Worker id #{workerid} processing fm: {factor_multiplier}')

    data_x, data_y, expls = data
    del data

    # create summaries - both
    csummary = tmexplainer.explanation_summaries(data_x, fm=factor_multiplier, precomputed_explanations=expls)
    ssummary = tmexplainer.simple_summaries(data_x)
    print(f'Worker id #{workerid} finished creating summaries for fm: {factor_multiplier}')

    csummary_texts = list(map(lambda x: x[0], csummary))
    ssummary_texts = list(ssummary)

    model = ''
    with FileLock(os.path.join(os.path.expanduser(modelpath), "iolock.lock")):
        model = eh.load_lstm_model(os.path.expanduser(modelpath))

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
    print(f'Worker id #{workerid} finished processing multiplier {factor_multiplier}')

    return out.getvalue()


@ray.remote
def precompute_explanations(data, cnames, modelpath, workerid):
    """
    data must be map of id:(string, int)
    """

    explanator = lime_text.LimeTextExplainer(class_names=cnames)

    # read model with filelock
    model = ''
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

    outdict = {}

    for uid, text in data.items():
        explanation = explanator.explain_instance(text[0], _predict_proba_fn, num_features=100)
        outdict[uid] = explanation.as_list()

    print(f"Worker @{workerid} precomputed {len(data.keys())} instances")
    return outdict


class FindFactorExperiment:

    def __init__(self, experiment_tag):
        self.experimenttag = experiment_tag
        self.language = "english"
        self.modelpath = "~/school/diploma/src/raw-data/lstm-model-sigmoid"  # path to model on all nodess

        now = time.strftime("%Y-%m-%d_%H:%M")
        self.experimentpath = f'/home/tomasmizera/school/diploma/src/data/experiments/{experiment_tag}'
        self.basepath = os.path.join(self.experimentpath, now)
        self.logpath = os.path.join(self.basepath, 'log')
        self.outpath = os.path.join(self.basepath, 'csv')

        # prepare output folder
        try:
            os.makedirs(self.basepath, exist_ok=True)
            os.makedirs(self.logpath, exist_ok=True)
            os.makedirs(self.outpath, exist_ok=True)

            latestpath = os.path.join(self.experimentpath, 'latest')
            if os.path.islink(latestpath):
                os.remove(latestpath)
            os.symlink(self.basepath, latestpath, target_is_directory=True)
        except FileExistsError as e:
            self._logw(e)
            sys.exit(1)

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
        data = eh.load_imdb_dataset('train+test')
        data = eh.preprocess_dataset(data, minnumofsentences)
        datagen = eh.generate_batches_of_size(data, batchsize)
        datachunk = ''

        try:
            datachunk = next(datagen)
        except StopIteration as e:  # Ran out of data, should not happen
            self._logw(e)
            return 1

        # convert tuple(list(x),list(y)) to dict(id: tuple(x, y))
        data_x, data_y = datachunk
        datadict = {}
        instanceid = 0
        for i, x in enumerate(data_x):
            datadict[instanceid] = (x, data_y[i])
            instanceid += 1

        self._logi(f'Data prepared for experiment {self.experimenttag}')
        return datadict

    def run(self, batch_size=100, to_factor=100, from_sentences_count=10):
        self._logw(f'Starting experiment - {self.experimenttag} ------------------------')

        start = time.time()
        ray.init(address='auto', _redis_password='5241590000000000')

        datachunk = self.prepare_data(batch_size, from_sentences_count)
        if datachunk == 1:
            return 1

        self._logi(f'Started precomputing explanations {self.experimenttag}')

        # precompute explanations
        precomputing_worker_ids = []
        wid = 0
        for x in eh.dict_to_chunks(datachunk, 5):
            eid = precompute_explanations.remote(
                x,
                ['Positive', 'Negative'],
                self.modelpath,
                wid
            )
            precomputing_worker_ids.append(eid)
            wid += 1

        progress = 0.25
        precomputed_data = ([], [], [])  # text, label, explanation
        while precomputing_worker_ids:
            done_ids, precomputing_worker_ids = ray.wait(precomputing_worker_ids)
            res = ray.get(done_ids[0])

            for instanceid, explanation in res.items():
                print(f"Saving precomputed id: {instanceid}")
                precomputed_data[0].append(datachunk[instanceid][0])  # instance text
                precomputed_data[1].append(datachunk[instanceid][1])  # instance label
                precomputed_data[2].append(explanation)  # instance explanation

            if (len(precomputed_data[0]) / batch_size) > progress:
                self._logi(f'Precomputed {progress*100}% of data')
                progress += 0.25

        assert len(datachunk) == len(precomputed_data[0])
        assert len(precomputed_data[0]) == len(precomputed_data[1]) == len(precomputed_data[2])
        del datachunk

        self._logi(f'Precomputed explanations for {self.experimenttag}, took: {time.time() - start}')

        # save precomputated data
        with open(os.path.join(self.basepath, "precomputed.pickle"), 'wb') as f:
            pickle.dump(precomputed_data, f)
            self._logi(f'Precomputed data saved to {self.basepath}')

        self._logi(f'Started creating summaries {self.experimenttag}')

        datachunkid = ray.put(precomputed_data)

        working_ids = []
        fm_mapping = {}  # mapping fm: ray worker id
        for factormultiplier in eh.generate_sequence(to_factor):
            task_id = initialize_task.remote(
                datachunkid,  # id of tuple (list of instances ~ strings, list of labels ~ ints)
                factormultiplier,
                self.modelpath,
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
