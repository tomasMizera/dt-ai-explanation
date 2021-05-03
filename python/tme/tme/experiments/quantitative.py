import psutil
from filelock import FileLock
from lime import lime_text
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
def compute_log_summary(
        factor_multiplier,
        precomputed_path,
        modelpath,
        outpath,
        workerid,
        sentences_count=6,
        explanation_filter=None):
    """
    Task for each worker with given fm, calculates and logs output
    """

    print(f'Worker id #{workerid} started processing fm: {factor_multiplier}, filter: {explanation_filter}')

    tmexplainer = tme.TextModelsExplainer()
    i = 0

    datagen = eh.read_precomputed_g(precomputed_path, 10)
    for chunk in datagen:
        tmp = list(zip(*chunk))  # list of tuples to tuple of lists
        if len(tmp) == 3:
            data_x, expls, data_y = tmp
        elif len(tmp) == 4:
            code, data_y, expls, data_x = tmp
        else:
            raise ValueError("Unable to unpack input data")

        data_x = list(data_x)
        data_y = list(data_y)
        expls = list(expls)

        if explanation_filter is not None:
            if type(explanation_filter) is int:  # filter by max number of explanation words
                expls = [x.as_list()[:explanation_filter] for x in expls]
            elif type(explanation_filter) is float:  # filter by minimum word weight
                expls = [list(filter(lambda w: w[1] > explanation_filter, x.as_list())) for x in expls]

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

        out = os.path.join(outpath, f'fm-{factor_multiplier}-{i}-{sentences_count}.csv')

        # explanation filter to filename
        if explanation_filter is not None:
            extag = explanation_filter
            if type(explanation_filter) is float:
                decimalplaces = 10 ** len(str(explanation_filter).split('.')[1])
                extag = str(int(explanation_filter * decimalplaces)) + f'eminus{decimalplaces}'
            out = os.path.join(outpath, f'fm-{factor_multiplier}-{i}-{sentences_count}-ef{extag}.csv')

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
def precompute_explanations(*, data, cnames, modeltype, modelpath, exp_filter, outpath, workerid):
    """
    data must be map of id:(string, int)
    data ~ list of tuples (instance, label)
    """

    explanator = lime_text.LimeTextExplainer(class_names=cnames)

    if modeltype == "lstm-imdb":
        with FileLock(os.path.join(modelpath, "iolock.lock")):
            model = eh.load_lstm_model(os.path.expanduser(modelpath))
    elif modeltype == "svm-lime":
        model, vectorizer = eh.load_religion_model(modelpath)
    else:
        raise ValueError("Unknown model! " + modeltype)

    def _predict_proba_fn(_input):
        """
        Function accepting array of instances and returns a probability for each class
        _input - 1d array of instances
        Returns 2d array of [num of instances] x [num of classes] with probabilities
        """
        if modeltype == "svm-lime":
            return model.predict_proba(vectorizer.transform(_input))
        elif modeltype == "lstm-imdb":
            prediction = model.predict(_input)
            return np.append(prediction, 1 - prediction, axis=1)

    assert (type(exp_filter) is int)

    out = []
    for text, label in data:
        explanation = explanator.explain_instance(text, _predict_proba_fn, num_features=exp_filter)
        out.append((text, explanation, label))

    outdir = os.path.join(outpath, f"expf:{exp_filter}")

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, f'filter:{exp_filter}-part:{workerid}.pickle'), 'wb') as fout:
        pickle.dump(out, fout)

    print(f"Worker @{workerid} precomputed {len(data)} instances")


class QuantitativeExperiment:

    def __init__(self, experiment_tag):
        self.experimenttag = experiment_tag
        self.language = "english"
        self.modelpath = ""

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

    def run_precompute(self, *,
                       instances_count=None,
                       dataset=None,
                       classes=None,
                       data_path=None,
                       explanation_filters=None,
                       modeltype=None,
                       min_sentences_count=None,
                       worker_load=None,
                       workers=None
                       ):
        self._logi(f'Started precomputing for {instances_count} instances with minimum {min_sentences_count} sentences')

        if dataset == "imdb":
            dataset = eh.load_imdb_dataset('train+test')
            datagen = eh.prepare_dataset_v2(
                dataset,
                min_sentences_count=min_sentences_count,
                batch_size=worker_load,
                expected_size=instances_count
            )
            del dataset
        elif dataset == "religion":
            datagen = eh.load_religion_dataset(data_path, batch_size=worker_load)
        else:
            raise ValueError("Unknown dataset " + dataset)

        i = 0
        workers_cnt = 0
        precomputing_worker_ids = []

        def _cond(x, y, z):
            if instances_count == "all":
                return True
            return x * y < z

        if type(explanation_filters) is not list:
            explanation_filters = [explanation_filters]

        while _cond(i, worker_load, instances_count):

            try:
                chunk = next(datagen)
            except StopIteration as e:  # No more data
                self._logw(f'Stopping, no more data, finished at iteration #{i}')
                break

            parts_counter = 0
            for filter_i in explanation_filters:
                # spawn worker for chunk
                precomputing_worker_ids.append(precompute_explanations.remote(
                    data=chunk,
                    cnames=classes,
                    modeltype=modeltype,
                    modelpath=self.modelpath,
                    exp_filter=filter_i,
                    outpath=self.datapath,
                    workerid=parts_counter
                ))
                self._logd(f'Scheduled worker @{workers_cnt}')
                parts_counter += 1
                workers_cnt += 1

                if workers_cnt % workers == 0:
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

    def spawn_summary_worker(self, fpm, precomputed_path, sentences_count=6, explanation_size=None):
        eid = compute_log_summary.remote(
            fpm,
            precomputed_path,
            self.modelpath,
            self.outpath,
            f'work@{fpm}',
            sentences_count=sentences_count,
            explanation_filter=explanation_size
        )
        return eid

    def run_summaries(self, **config):
        self._logi(f'Started creating summaries {self.experimenttag}')

        working_ids = []
        it = 0
        cores = config["workers"]

        seq = [config["factor"]]
        if config["run_to_factor"]:
            seq = eh.generate_sequence(config["factor"])

        senlen = config["summary_length"]
        if type(config["summary_length"]) != list:
            senlen = [senlen]

        explsize = config["explanation_filter"]
        if type(explsize) != list:
            explsize = [explsize]

        for factormultiplier in seq:
            for sentences_l in senlen:
                for explsize_l in explsize:
                    working_ids.append(self.spawn_summary_worker(
                        factormultiplier,
                        config["precomputed_dir"],
                        sentences_count=sentences_l,
                        explanation_size=explsize_l
                    ))

                    if (it + 1) % cores == 0:
                        self._logd(f'Waiting to sync at @{it}')
                        done, working_ids = ray.wait(working_ids, num_returns=len(working_ids))
                        del done
                        self._logd(f'Synced, restarting ray @{it}')
                        ray.shutdown()
                        ray.init(address='auto', _redis_password='5241590000000000')

                    it += 1

                    self._logd(f'Scheduled work for fm {factormultiplier}, length {sentences_l} and explf {explsize_l}')

        # save done tasks (other workers might still be working)
        while working_ids:
            done_id, working_ids = ray.wait(working_ids)
            del done_id
            self._logd(f'Worker finished with summaries')

    def run(self, precompute=True, **config):
        self._logw(f'Starting experiment - {self.experimenttag} ------------------------')

        tagline = "PRECOMPUTE" if precompute else "QUANTITATIVE" if not config['summarizing']['run_to_factor'] \
            else "FINDING MAX"
        self._logi(f'Experiment setup:\n- {tagline}\n- config: {config}\n')

        self.modelpath = config["model_path"]

        start = time.time()
        ray.init(address='auto', _redis_password='5241590000000000')

        if precompute:
            self.run_precompute(
                instances_count=config["precomputing"]["instances"],
                dataset=config["dataset"],
                classes=config["classes"],
                data_path=config["precomputing"]["data_path"],
                explanation_filters=config["precomputing"]["explanation_filter"],
                modeltype=config["model"],
                min_sentences_count=config["precomputing"]["min_sentences_count"],
                worker_load=config["precomputing"]["worker_load"],
                workers=config["precomputing"]["workers"]
            )
            self._logi(f'Precomputed explanations for {self.experimenttag}, took: {time.time() - start}')

        if not precompute:
            self.run_summaries(**config['summarizing'])
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
