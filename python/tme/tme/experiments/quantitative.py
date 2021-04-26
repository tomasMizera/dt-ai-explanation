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
def initialize_task(factor_multiplier, precomputed_path, modelpath, outpath, workerid):
    """
    Task for each worker with given fm, calculates and logs output
    """

    print(f'Worker id #{workerid} started processing fm: {factor_multiplier}')

    tmexplainer = tme.TextModelsExplainer()
    i = 0

    datagen = eh.read_precomputed_g(precomputed_path, 100)
    for chunk in datagen:
        data_x, expls, data_y = zip(*chunk)  # list of tuples to tuple of lists
        data_x = list(data_x)
        data_y = list(data_y)
        expls = list(expls)

        # create summaries - both
        csummary = tmexplainer.explanation_summaries(data_x, fm=factor_multiplier, precomputed_explanations=expls)
        ssummary = tmexplainer.simple_summaries(data_x)

        csummary = list(map(lambda x: x[0], csummary))
        ssummary = list(ssummary)

        with FileLock(os.path.join(os.path.expanduser(modelpath), "iolock.lock")):
            model = eh.load_lstm_model(os.path.expanduser(modelpath))

        csummary = model.predict(csummary)
        ssummary = model.predict(ssummary)
        modelp = model.predict(data_x)

        assert (all([len(modelp) == len(ssummary),
                     len(ssummary) == len(csummary)]))

        # convert true labels
        labels = np.array(list(map(lambda x: [x], data_y)))

        modelp = np.append(modelp, csummary, axis=1)
        modelp = np.append(modelp, ssummary, axis=1)
        modelp = np.append(modelp, labels, axis=1)

        out = os.path.join(outpath, f'fm-{factor_multiplier}-{i}.csv')
        np.savetxt(
            out,
            modelp,
            fmt='%1.5f',
            header="originalP,customSP,simpleSP,trueClass",
            comments='',
            delimiter=','
        )
        i += 1

    print(f'Worker id #{workerid} finished processing multiplier {factor_multiplier}')


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

            if i and i % 8 == 0:
                self._logd(f'Waiting to sync at @{i}')
                done, precomputing_worker_ids = ray.wait(
                    precomputing_worker_ids,
                    num_returns=len(precomputing_worker_ids)
                )
                del done
                self._logd(f'Synced, restarting ray @{i}')
                ray.shutdown()
                ray.init(address='auto', _redis_password='5241590000000000')

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

    def run_summaries(self, factor, run_to_factor, precomputed_path, workers):
        self._logi(f'Started creating summaries {self.experimenttag}')

        working_ids = []
        it = 0
        cores = workers

        seq = [factor]
        if run_to_factor:
            seq = eh.generate_sequence(factor)

        for factormultiplier in seq:
            working_ids.append(initialize_task.remote(
                factormultiplier,
                precomputed_path,
                self.modelpath,
                self.outpath,
                f'work@{factormultiplier}'
            ))
            self._logd(f'Scheduled work for fm {factormultiplier}')

            if (it + 1) % cores == 0:
                self._logd(f'Waiting to sync at @{it}')
                done, working_ids = ray.wait(working_ids, num_returns=len(working_ids))
                del done
                self._logd(f'Synced, restarting ray @{it}')
                ray.shutdown()
                ray.init(address='auto', _redis_password='5241590000000000')

            it += 1

        # save done tasks (other workers might still be working)
        while working_ids:
            done_id, working_ids = ray.wait(working_ids)
            del done_id
            self._logd(f'Worker finished with summaries')

    def run(self,
            instances=100,
            factor=100,
            from_sentences_count=10,
            run_to_factor=False,
            precompute=True,
            worker_load=25,
            path_to_precomputed='',
            workers=8):
        self._logw(f'Starting experiment - {self.experimenttag} ------------------------')

        tagline = "PRECOMPUTE" if precompute else "QUANTITATIVE" if not run_to_factor else "FINDING MAX"
        if precompute:
            self._logi(f'Experiment setup:\n- {tagline}\n'
                       f'- min sentences count: {from_sentences_count}\n'
                       f'- # of instances: {instances}\n'
                       f'- work load: {worker_load}')
        else:
            self._logi(f'Experiment setup:\n- {tagline}\n'
                       f'- factor: {factor}\n'
                       f'- path to precomputed: {path_to_precomputed}')

        start = time.time()
        ray.init(address='auto', _redis_password='5241590000000000')

        if precompute:
            self.run_precompute(instances, from_sentences_count, worker_load)
            self._logi(f'Precomputed explanations for {self.experimenttag}, took: {time.time() - start}')

        if not precompute:
            self.run_summaries(factor, run_to_factor, path_to_precomputed, workers)
            self._logi(f'Data written to {self.outpath}')

        end = time.time()

        self._logi(f'Experiment {self.experimenttag} took {end - start}')
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
