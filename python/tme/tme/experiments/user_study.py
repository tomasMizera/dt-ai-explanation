import ray
import pickle
import os
from . import experiment_helper as eh
import time
from lime import lime_text


@ray.remote
def precompute_explanations(data, cnames, modelpath, vectorizerpath, outpath, label, workerid):

    explanator = lime_text.LimeTextExplainer(class_names=cnames)
    print(f'Worker @{workerid} started operating')

    model = eh.load_pickle_object(modelpath)
    assert(model is not None)

    vectorizer = eh.load_pickle_object(vectorizerpath)
    assert(vectorizer is not None)

    def _predict_proba_fn(_input):
        """
        Function accepting array of instances and returns a probability for each class
        _input - 1d array of instances
        Returns 2d array of [num of instances] x [num of classes] with probabilities
        """
        return model.predict_proba(vectorizer.transform(_input))

    out = []
    for i in range(len(data)):
        explanation = explanator.explain_instance(data[i][0], _predict_proba_fn, num_features=100)
        out.append((label, data[i][1], explanation, data[i][0]))

    with open(os.path.join(outpath, f'{workerid}.pickle'), 'wb') as fout:
        pickle.dump(out, fout)

    print(f"Worker @{workerid} precomputed {len(data)} instances with label {label}")
    return workerid


def run_workers(data, modelpath, vectorizerpath, outpath, workers, label):
    classnames = ["atheism", "christianity"]
    batch_size = 50
    start = time.time()
    print(f'Started precomputing label {label} at {start}')

    index = 0
    wids = []
    for offset in range(0, len(data), batch_size):
        wids.append(precompute_explanations.remote(
            data[offset: offset+batch_size],
            classnames,
            modelpath,
            vectorizerpath,
            outpath,
            label,
            f'{label}-{index}'
        ))

        if (index + 1) % workers == 0:  # do not push new tasks when there are no free workers
            print(f'{time.time() - start}: Waiting to sync at @{index}')
            done, wids = ray.wait(
                wids,
                num_returns=len(wids)
            )
            del done
            print(f'{time.time()-start}:  Synced, restarting ray @{index}')
            ray.shutdown()
            ray.init(num_cpus=workers)

        index += 1

    # wait for the rest of the workers
    while wids:
        uid, wids = ray.wait(wids)
        print(f'Worker @{ray.get(uid[0])} finished')
        del uid

    print(f'{time.time() - start}: Precomputed {label}')


def run(datapath, modelpath, vectorizerpath, outpath, workers):

    ray.init(num_cpus=workers)

    data_christianity = eh.load_files(os.path.join(datapath, "christianity"))
    data_atheism = eh.load_files(os.path.join(datapath, "atheism"))

    data_christianity = list(zip(data_christianity, [1] * len(data_christianity)))
    data_atheism = list(zip(data_atheism, [0] * len(data_atheism)))

    run_workers(data_christianity, modelpath, vectorizerpath, outpath, workers, 'c')
    run_workers(data_atheism, modelpath, vectorizerpath, outpath, workers, 'a')

    time.sleep(10)  # give some time to join workers
    ray.shutdown()
    print(f'Precomputing finished')
