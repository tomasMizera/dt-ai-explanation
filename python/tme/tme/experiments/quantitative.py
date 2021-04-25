import psutil
from filelock import FileLock
from lime import lime_text
from io import StringIO
import numpy as np
import logging
import pickle
import time
import ray
import sys
import os
import gc

import tme.experiments.experiment_helper as eh
from tme.src import tme


@ray.remote
def initialize_task(data, factor_multiplier, modelpath, workerid):
    """
    Task for each worker with given fm, calculates and logs output
    """

    print(f'Worker id #{workerid} started operating')

    tmexplainer = tme.TextModelsExplainer()

    print(f'Worker id #{workerid} processing fm: {factor_multiplier}')

    data_x, data_y, expls = data
    del data

    # create summaries - both
    csummary = tmexplainer.explanation_summaries(data_x, fm=factor_multiplier, precomputed_explanations=expls)
    ssummary = tmexplainer.simple_summaries(data_x)
    print(f'Worker id #{workerid} finished creating summaries for fm: {factor_multiplier}')

    csummary_texts = list(map(lambda x: x[0], csummary))
    ssummary_texts = list(ssummary)

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
def precompute_explanations(data, cnames, modelpath, outpath, workerid):
    """
    data must be map of id:(string, int)
    data ~ list of tuples (instance, label)
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

    out = []

    for text, label in data:
        explanation = explanator.explain_instance(text, _predict_proba_fn, num_features=100)
        out.append((text, explanation, label))

    with open(os.path.join(outpath, f'{workerid}.pickle'), 'wb') as fout:
        pickle.dump(out, fout)

    print(f"Worker @{workerid} precomputed {len(data)} instances")
    return workerid


class QuantitativeExperiment:

    def __init__(self, experiment_tag):
        self.experimenttag = experiment_tag
        self.language = "english"
        self.modelpath = "~/school/diploma/src/raw-data/lstm-model-sigmoid"  # path to model on all nodess

        now = time.strftime("%Y-%m-%d_%H:%M")
        self.experimentpath = f'/home/tomasmizera/school/diploma/src/data/experiments/{experiment_tag}'
        self.basepath = os.path.join(self.experimentpath, now)
        self.logpath = os.path.join(self.basepath, 'log')
        self.outpath = os.path.join(self.basepath, 'csv')
        self.datapath = os.path.join(self.basepath, 'precomputed')

        logfilepath = os.path.join(self.logpath, f'{self.experimenttag}-{now}.log')

        # prepare output folder
        try:
            os.makedirs(self.basepath, exist_ok=True)
            os.makedirs(self.logpath, exist_ok=True)
            os.makedirs(self.outpath, exist_ok=True)
            os.makedirs(self.datapath, exist_ok=True)

            latestpath = os.path.join(self.experimentpath, 'latest')
            latestlogpath = os.path.join(self.logpath, 'latest.log')
            if os.path.islink(latestpath):
                os.remove(latestpath)
            if os.path.islink(latestlogpath):
                os.remove(latestlogpath)
            os.symlink(self.basepath, latestpath, target_is_directory=True)
            os.symlink(logfilepath, latestlogpath)
        except FileExistsError as e:
            self._logw(e)
            sys.exit(1)

        # setup logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(filename=logfilepath)
            ]
        )

        self.log = logging.getLogger()

    def run_precompute(self, instances_count=10, min_sentences_count=30, worker_load=25):
        self._logi(f'Started precomputing for {instances_count} instances with minimum {min_sentences_count} sentences')

        dataset = eh.load_imdb_dataset('train+test')
        datagen = eh.prepare_dataset_v2(
            dataset,
            min_sentences_count=min_sentences_count,
            batch_size=worker_load,
            expected_size=instances_count
        )
        del dataset

        i = 0
        precomputing_worker_ids = []

        while i * worker_load < instances_count:

            try:
                chunk = next(datagen)
            except StopIteration as e:  # No more data
                self._logw(f'Stopping, no more data, finished at iteration #{i}')
                break

            # spawn worker for chunk
            precomputing_worker_ids.append(precompute_explanations.remote(
                chunk,
                ['Positive', 'Negative'],
                self.modelpath,
                self.datapath,
                i
            ))
            self._logd(f'Scheduled worker @{i}')

            i += 1

        # wait for the rest of the workers
        while precomputing_worker_ids:
            uid, precomputing_worker_ids = ray.wait(precomputing_worker_ids)
            self._logd(f'Worker @{ray.get(uid)} finished')
            del uid

            if psutil.virtual_memory().percent > 80:
                col = gc.collect()
                self._logd('GC collected ' + str(col))

        self._logi(f'Finished precomputing')

    def run_summaries(self, factor, run_to_factor):
        self._logi(f'Started creating summaries {self.experimenttag}')

        precomputed_data = eh.read_precomputed_g('')
        datachunkid = ray.put(precomputed_data)

        working_ids = []
        fm_mapping = {}  # mapping fm: ray worker id

        seq = [factor]
        if run_to_factor:
            seq = eh.generate_sequence(factor)

        for factormultiplier in []:  # !! replace with seq
            task_id = initialize_task.remote(
                datachunkid,  # id of tuple (list of instances ~ strings, list of labels ~ ints)
                factormultiplier,
                self.modelpath,
                f'work@{factormultiplier}'
            )
            fm_mapping[task_id] = factormultiplier
            working_ids.append(task_id)

        try:
            del task_id
        except NameError:
            pass

        # save done tasks (other workers might still be working)
        while working_ids:
            done_id, working_ids = ray.wait(working_ids)
            done_id = done_id[0]
            done_fm = fm_mapping[done_id]

            outfilepath = os.path.join(self.outpath, f'fm-{done_fm}.csv')
            with open(outfilepath, 'w') as f:
                f.write(ray.get(done_id))
                self._logd(f'Saved data for fm {done_fm} to {outfilepath}')

            del fm_mapping[done_id], done_id

        del datachunkid

    def run(self,
            instances=100,
            factor=100,
            from_sentences_count=10,
            run_to_factor=False,
            only_precompute=True,
            batch_size=25):
        self._logw(f'Starting experiment - {self.experimenttag} ------------------------')
        tagline = "PRECOMPUTE" if only_precompute else "QUANTITATIVE" if not run_to_factor else "FINDING MAX"
        self._logi(f'Experiment setup:\n- {tagline}\n'
                   f'- factor / to factor: {factor}\n'
                   f'- min sentences count: {from_sentences_count}\n'
                   f'- # of instances: {instances}')

        start = time.time()
        ray.init(address='auto', _redis_password='5241590000000000')

        self.run_precompute(instances, from_sentences_count, batch_size)

        self._logi(f'Precomputed explanations for {self.experimenttag}, took: {time.time() - start}')

        if not only_precompute:
            self.run_summaries(factor, run_to_factor)

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
